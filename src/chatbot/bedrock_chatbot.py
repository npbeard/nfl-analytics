"""
NFL Analytics NLP Chatbot — powered by Amazon Bedrock (Claude).

Provides a conversational interface over NFL stats and injury data.
Integrates with the injury-prediction endpoint to answer questions like:
  "What is the injury risk for a running back on synthetic turf in cold weather?"
  "Show me the top 5 plays with the highest injury severity last season."

Deployment on AWS:
  - Bedrock client: invoke_model / converse API with claude-3-5-sonnet
  - Conversation history persisted in DynamoDB (session management)
  - Exposed via API Gateway + Lambda (REST or WebSocket for streaming)
  - Optional: Bedrock Agents with tool use for structured data queries
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

BEDROCK_MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"

SYSTEM_PROMPT = """You are the NFL Analytics Assistant, an expert on American football data,
player performance, and injury analytics. You have access to:
- Play-by-play data for all NFL games (PlayList dataset)
- Injury records with severity classifications (InjuryRecord dataset)
- Real-time player tracking from the Next Gen Stats system

You help coaches, analysts, and medical staff understand trends, query data,
and interpret injury risk predictions. Be concise, data-driven, and reference
specific statistics when available. Always flag if a response involves medical
decisions that require professional evaluation."""


@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str


@dataclass
class ChatSession:
    session_id: str
    history: list[Message] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    def add(self, role: str, content: str) -> None:
        self.history.append(Message(role=role, content=content))

    def to_bedrock_messages(self) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in self.history]


class NFLChatbot:
    """
    Conversational chatbot backed by Amazon Bedrock.
    Falls back to a mock response when boto3/Bedrock is unavailable (local dev).
    """

    def __init__(self, model_id: str = BEDROCK_MODEL_ID, use_mock: bool = False):
        self.model_id = model_id
        self.use_mock = use_mock
        self._client = None

        if not use_mock:
            try:
                import boto3
                self._client = boto3.client("bedrock-runtime", region_name="eu-west-1")
            except ImportError:
                logger.warning("boto3 not available — falling back to mock mode")
                self.use_mock = True

    def chat(self, session: ChatSession, user_message: str) -> str:
        session.add("user", user_message)

        if self.use_mock:
            response = self._mock_response(user_message)
        else:
            response = self._invoke_bedrock(session)

        session.add("assistant", response)
        return response

    def _invoke_bedrock(self, session: ChatSession) -> str:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "system": SYSTEM_PROMPT,
            "messages": session.to_bedrock_messages(),
        }
        resp = self._client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(resp["body"].read())
        return result["content"][0]["text"]

    def _mock_response(self, user_message: str) -> str:
        """Deterministic stub for local testing without AWS credentials."""
        lower = user_message.lower()
        if "injury" in lower and "synthetic" in lower:
            return (
                "Based on the InjuryRecord dataset, injuries on synthetic surfaces account for "
                "roughly 55% of recorded incidents despite synthetic fields representing only ~40% "
                "of game snaps. Knee and ankle injuries are most prevalent on synthetic turf, "
                "particularly for skill positions (RB, WR). The injury prediction model estimates "
                "a ~12% higher DM_M7 probability (7+ days missed) on synthetic vs natural grass."
            )
        if "top" in lower and ("play" in lower or "risk" in lower):
            return (
                "The highest injury-risk scenarios from the training data are: "
                "(1) Running backs on synthetic turf in outdoor cold-weather stadiums, "
                "(2) Pass plays in games beyond PlayerDay 80 (late-season fatigue), "
                "(3) Players with >35 snaps in the current game (PlayerGamePlay feature). "
                "These factors are the top contributors in the Random Forest feature importance."
            )
        if "architecture" in lower or "aws" in lower:
            return (
                "The proposed AWS architecture routes live NGS sensor data through "
                "Kinesis Data Streams → Kinesis Data Analytics (Flink) for real-time stats. "
                "Batch data lands in S3 (raw zone), is transformed by Glue jobs into the "
                "processed zone, and loaded into Redshift for BI queries. "
                "SageMaker hosts the injury prediction endpoint. "
                "This chatbot is served via Bedrock + API Gateway."
            )
        return (
            f"I received your question about: \"{user_message}\". "
            "In production I would query the Redshift analytics layer and the "
            "SageMaker inference endpoint to give you a data-backed answer. "
            "Please connect to AWS to enable live responses."
        )


def build_tool_definitions() -> list[dict]:
    """
    Bedrock Agents tool definitions for structured data queries.
    These let the model call back into the analytics pipeline.
    """
    return [
        {
            "name": "get_injury_risk",
            "description": "Predict injury severity probabilities for given play conditions.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "surface": {"type": "string", "enum": ["Natural", "Synthetic"]},
                    "stadium_type": {"type": "string", "enum": ["Outdoor", "Indoors"]},
                    "position_group": {"type": "string"},
                    "temperature": {"type": "number"},
                    "player_day": {"type": "integer"},
                    "play_type": {"type": "string", "enum": ["Pass", "Rush", "Kickoff"]},
                },
                "required": ["surface", "play_type"],
            },
        },
        {
            "name": "query_play_stats",
            "description": "Return aggregated play statistics filtered by game conditions.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "field_type": {"type": "string"},
                    "weather": {"type": "string"},
                    "position_group": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": [],
            },
        },
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    bot = NFLChatbot(use_mock=True)
    session = ChatSession(session_id="demo-001")

    questions = [
        "What is the injury risk for a running back on synthetic turf?",
        "Which play scenarios carry the highest risk of 7+ days missed?",
        "Can you explain the AWS architecture used for this platform?",
    ]

    print("NFL Analytics Chatbot — Demo (mock mode)\n" + "=" * 50)
    for q in questions:
        print(f"\nUser: {q}")
        answer = bot.chat(session, q)
        print(f"Assistant: {answer}")

    print(f"\nSession history: {len(session.history)} messages")

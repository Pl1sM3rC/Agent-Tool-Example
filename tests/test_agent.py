from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import agent


class FakeToolCallingModel(GenericFakeChatModel):
    """Replays scripted messages; create_agent needs bind_tools, which the fake model doesn't implement."""

    def bind_tools(self, tools, **kwargs):
        return self


async def test_agent_calls_mcp_tool_and_answers(monkeypatch, capsys):
    model = FakeToolCallingModel(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "count_r", "args": {"word": "Strawberry"}, "id": "call_1"}]),
        AIMessage(content="Strawberry has 3 r's."),
    ]))
    monkeypatch.setattr(agent, "create_model", lambda: model)

    await agent.main("How many r's does the word Strawberry have?")

    out = capsys.readouterr().out
    assert "Tool Message" in out and "count_r" in out
    assert "'text': '3'" in out  # the tool result from the MCP server (as content blocks)
    assert "Strawberry has 3 r's." in out
    assert "--- Graph Async Stream End ---" in out


def test_create_model_defaults_to_deepseek_with_thinking_disabled(monkeypatch):
    for var in ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    model = agent.create_model()
    assert model.openai_api_base == "https://api.deepseek.com"
    assert model.model_name == "deepseek-flash"
    assert model.extra_body == {"thinking": {"type": "disabled"}}


def test_create_model_uses_endpoint_from_env(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("LLM_MODEL", "some-local-model")
    model = agent.create_model()
    assert model.openai_api_base == "http://localhost:1234/v1"
    assert model.model_name == "some-local-model"
    assert model.extra_body is None

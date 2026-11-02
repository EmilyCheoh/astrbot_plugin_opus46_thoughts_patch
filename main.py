"""Extract explicitly tagged thoughts from completed AstrBot agent replies.

Protocol (literal braces, not XML):
    {thoughts}
    text for a separate forwarded message
    {/thoughts}
    normal answer

Designed for non-streaming AstrBot v4.24.5. No model filter.
"""

from dataclasses import dataclass

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Node, Plain
from astrbot.api.star import Context, Star, register
from astrbot.core.agent.message import Message, TextPart
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.astr_agent_context import AstrAgentContext
from astrbot.api.provider import LLMResponse

OPEN = "{thoughts}"
CLOSE = "{/thoughts}"


@dataclass
class _TextSlot:
    """A writable piece of assistant text in the current in-memory history."""

    message: Message
    part: TextPart | None
    original: str

    def write(self, value: str) -> None:
        if self.part is None:
            self.message.content = value
        else:
            self.part.text = value


def _slots_after_latest_user(messages: list[Message]) -> list[_TextSlot]:
    """Only inspect assistants in the current user turn, never old chat turns."""
    last_user = -1
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].role == "user":
            last_user = idx
            break

    slots: list[_TextSlot] = []
    for msg in messages[last_user + 1 :]:
        if msg.role != "assistant":
            continue
        if isinstance(msg.content, str):
            slots.append(_TextSlot(msg, None, msg.content))
        elif isinstance(msg.content, list):
            for part in msg.content:
                if isinstance(part, TextPart):
                    slots.append(_TextSlot(msg, part, part.text))
    return slots


def _extract(slots: list[_TextSlot]) -> tuple[str, list[tuple[_TextSlot, str]]] | None:
    """Plan changes; do not mutate anything until the forwarded message is sent.

    A tagged block may span assistant messages separated by tool calls. Tool
    messages, tool_call metadata, ThinkPart, images, etc. are never touched.
    Only an assistant text *beginning* with OPEN (apart from whitespace) can
    start a block. An unfinished block is left entirely unchanged.
    """
    found_open = False
    thought_parts: list[str] = []
    changes: list[tuple[_TextSlot, str]] = []

    for slot in slots:
        value = slot.original
        if not found_open:
            pos = len(value) - len(value.lstrip())
            if not value.startswith(OPEN, pos):
                continue
            found_open = True
            value = value[pos + len(OPEN) :]

        end = value.find(CLOSE)
        if end >= 0:
            thought_parts.append(value[:end])
            changes.append((slot, value[end + len(CLOSE) :]))
            # An empty thought is allowed: just strip the marker pair.
            return "\n".join(s.strip("\r\n") for s in thought_parts).strip(), changes

        thought_parts.append(value)
        changes.append((slot, ""))

    # No complete pair: leave the reply and history as written.
    return None


@register(
    "astrbot_plugin_opus46_thoughts_patch",
    "Felis Abyssalis & Abyss AI",
    "Extract {thoughts}...{/thoughts} into a separate message and clean history",
    "1.0.0",
)
class Opus46ThoughtsPatch(Star):
    def __init__(self, context: Context):
        super().__init__(context)

    @filter.on_agent_done()
    async def extract_thoughts(
        self,
        event: AstrMessageEvent,
        run_context: ContextWrapper[AstrAgentContext],
        response: LLMResponse,
    ) -> None:
        if response is None or response.role != "assistant":
            return

        messages = run_context.messages
        if not messages or messages[-1].role != "assistant":
            return

        slots = _slots_after_latest_user(messages)
        plan = _extract(slots)
        if plan is None:
            return

        thoughts, changes = plan
        if thoughts:
            try:
                await event.send(
                    event.chain_result(
                        [
                            Node(
                                uin="0",
                                name="🌬️ Abyss AI's thoughts",
                                content=[Plain(thoughts)],
                            )
                        ]
                    )
                )
            except Exception:
                # If delivery fails, preserve the original reply/history rather
                # than silently discarding the thought text.
                logger.exception("opus46_thoughts_patch: forwarding thoughts failed")
                return

        # Identify the final response's text pieces before mutating them.
        last_message = messages[-1]
        final_slots = [slot for slot in slots if slot.message is last_message]
        final_changed = any(slot.message is last_message for slot, _ in changes)
        cleaned_final = None
        if final_changed:
            replacement = {id(slot): text for slot, text in changes}
            cleaned_final = "".join(replacement.get(id(s), s.original) for s in final_slots)

        for slot, cleaned in changes:
            slot.write(cleaned)

        if final_changed and cleaned_final is not None:
            # Update the object that AstrBot subsequently uses for output.
            response.completion_text = cleaned_final

        logger.info("opus46_thoughts_patch: extracted tagged thoughts; cleaned runtime history")

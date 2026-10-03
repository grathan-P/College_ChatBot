
"""Conversation memory utilities for follow-up questions."""

from dataclasses import dataclass, field


@dataclass
class ConversationMemory:
    """Store recent conversation turns for follow-up questions."""

    max_turns: int = 5
    _messages: list[tuple[str, str]] = field(default_factory=list)

    def add_user_message(self, message: str) -> None:
        """Add a student message to memory."""
        message = message.strip()

        if message:
            self._messages.append(("user", message))
            self._trim()

    def add_assistant_message(self, message: str) -> None:
        """Add an assistant response to memory."""
        message = message.strip()

        if message:
            self._messages.append(("assistant", message))
            self._trim()

    def get_messages(self) -> list[tuple[str, str]]:
        """Return a copy of the stored conversation."""
        return self._messages.copy()

    def clear(self) -> None:
        """Clear the conversation history."""
        self._messages.clear()

    def _trim(self) -> None:
        """Keep only the configured number of recent turns."""
        max_messages = self.max_turns * 2

        if len(self._messages) > max_messages:
            self._messages = self._messages[-max_messages:]

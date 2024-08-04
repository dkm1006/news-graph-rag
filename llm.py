# Taken from https://github.com/b-art-b/langchain-snowpoc/tree/main/langchain_snowpoc
# Copyright lies with the initial creator, MIT licence does not apply to this part of the source

import logging
from typing import Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.messages import AIMessage
from snowflake import snowpark
from snowflake.connector.connection import SnowflakeConnection
from snowflake.cortex import Complete

logger = logging.getLogger(__name__)


class Cortex(LLM):
    connection: SnowflakeConnection = None

    model: str = "mistral-7b"

    @property
    def _llm_type(self) -> str:
        return "cortex"

    def _call(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        res = Complete(
            self.model,
            prompt,
            snowpark.Session.builder.configs({"connection": self.connection}).create(),
        )
        return res

    @property
    def _identifying_params(self) -> dict[str, SnowflakeConnection|str]:
        """Get the identifying parameters."""
        return {"connection": self.connection, "model": self.model}


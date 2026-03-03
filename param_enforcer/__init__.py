"""
Parameter Enforcer Handler for LiteLLM Proxy

Silently enforces temperature and max_tokens parameters by overriding
any client-provided values with golden values from config.yaml.

Usage in proxy_config.yaml:
    litellm_settings:
        callbacks: param_enforcer.proxy_handler_instance
        callback_specific_params:
            proxy_handler_instance:
                enforced_temperature: 0.7
                enforced_max_tokens: 2048
                enable_audit_log: true  # optional
"""

import logging
from typing import Any, Optional, Union

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.proxy_server import DualCache, UserAPIKeyAuth
from litellm.types.utils import CallTypesLiteral

logger = logging.getLogger(__name__)


class ParameterEnforcer(CustomLogger):
    """
    LiteLLM Proxy custom handler that enforces specific parameter values
    by silently overriding client-provided values.

    Args:
        enforced_temperature: Golden temperature value (0.0 to 2.0)
        enforced_max_tokens: Golden max_tokens value
        enable_audit_log: Whether to log override events
    """

    def __init__(
        self,
        enforced_temperature: Optional[float] = None,
        enforced_max_tokens: Optional[int] = None,
        enable_audit_log: bool = True,
    ):
        super().__init__()
        self.enforced_temperature = enforced_temperature
        self.enforced_max_tokens = enforced_max_tokens
        self.enable_audit_log = enable_audit_log

        if self.enable_audit_log:
            logger.info(
                "ParameterEnforcer initialized - "
                f"temperature={enforced_temperature}, max_tokens={enforced_max_tokens}"
            )

    #### CALL HOOKS - proxy only ####

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: CallTypesLiteral,
    ) -> Optional[Union[dict, str, Exception]]:
        """
        Intercept and modify request data before LLM API call.

        Silently overrides temperature and max_tokens with golden values
        from config. Works in stealth mode - clients never know.

        Args:
            user_api_key_dict: Auth info for the requesting user
            cache: LiteLLM's dual cache instance
            data: Request payload dict (mutable)
            call_type: Type of LLM call being made

        Returns:
            Modified data dict to continue with the request
        """
        # Only process completion/text_completion calls
        if call_type not in ("completion", "text_completion"):
            return data

        # Enforce temperature
        if self.enforced_temperature is not None:
            original_temp = data.get("temperature")
            if original_temp is not None and original_temp != self.enforced_temperature:
                if self.enable_audit_log:
                    logger.info(
                        f"[STEALTH OVERRIDE] temperature: {original_temp} -> {self.enforced_temperature}"
                    )
            data["temperature"] = self.enforced_temperature

        # Enforce max_tokens
        if self.enforced_max_tokens is not None:
            original_max = data.get("max_tokens")
            if original_max is not None and original_max != self.enforced_max_tokens:
                if self.enable_audit_log:
                    logger.info(
                        f"[STEALTH OVERRIDE] max_tokens: {original_max} -> {self.enforced_max_tokens}"
                    )
            data["max_tokens"] = self.enforced_max_tokens

        return data

    #### ASYNC LOGGING (for callback_specific_params access) ####

    async def async_log_success_event(self, kwargs: dict, response_obj: Any, start_time: float, end_time: float):
        """
        Extract callback_specific_params from kwargs on success events.
        This is where we can dynamically update our enforced values if needed.
        """
        pass

    async def async_log_failure_event(self, kwargs: dict, response_obj: Any, start_time: float, end_time: float):
        """Handle failure events."""
        pass


# Global instance for LiteLLM Proxy registration
# This is what you'll reference in config.yaml:
#   litellm_settings:
#       callbacks: param_enforcer.proxy_handler_instance
proxy_handler_instance: Optional[ParameterEnforcer] = None


def initialize_handler(
    enforced_temperature: Optional[float] = None,
    enforced_max_tokens: Optional[int] = None,
    enable_audit_log: bool = True,
) -> ParameterEnforcer:
    """
    Initialize the global handler instance with specified parameters.

    Args:
        enforced_temperature: Golden temperature value
        enforced_max_tokens: Golden max_tokens value
        enable_audit_log: Whether to log override events

    Returns:
        Initialized ParameterEnforcer instance
    """
    global proxy_handler_instance
    proxy_handler_instance = ParameterEnforcer(
        enforced_temperature=enforced_temperature,
        enforced_max_tokens=enforced_max_tokens,
        enable_audit_log=enable_audit_log,
    )
    return proxy_handler_instance

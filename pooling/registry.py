"""
Registry for pooling algorithms.

Allows registration of built-in and external poolers by name.
"""

from typing import Dict, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import CallPooler


class PoolerRegistry:
    """
    Registry for pooling algorithms.

    Allows poolers to be registered by name and retrieved later.
    Supports both decorator-based registration (for built-in poolers)
    and runtime registration (for external/plugin poolers).

    Example:
        # Decorator registration (in module)
        @PoolerRegistry.register("my_pooler")
        class MyPooler(CallPooler):
            ...

        # Runtime registration (for plugins)
        from external_lib import AdvancedPooler
        PoolerRegistry.register_external("advanced", AdvancedPooler)

        # Usage
        pooler = PoolerRegistry.get("average", n_features=11)
    """

    _poolers: Dict[str, Type['CallPooler']] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a pooler class.

        Args:
            name: Name to register the pooler under. Should be lowercase
                  and descriptive (e.g., "average", "attention").

        Returns:
            Decorator function that registers and returns the class.

        Example:
            @PoolerRegistry.register("average")
            class AveragePooler(CallPooler):
                ...
        """
        def decorator(pooler_cls: Type['CallPooler']) -> Type['CallPooler']:
            cls._poolers[name] = pooler_cls
            return pooler_cls
        return decorator

    @classmethod
    def get(cls, name: str, **kwargs) -> 'CallPooler':
        """
        Get a pooler instance by name.

        Args:
            name: Registered name of the pooler.
            **kwargs: Arguments passed to the pooler constructor.

        Returns:
            Instantiated pooler object.

        Raises:
            KeyError: If no pooler is registered with the given name.

        Example:
            pooler = PoolerRegistry.get("average", n_features=11)
        """
        if name not in cls._poolers:
            available = list(cls._poolers.keys())
            raise KeyError(
                f"Unknown pooler: '{name}'. Available: {available}"
            )
        return cls._poolers[name](**kwargs)

    @classmethod
    def register_external(cls, name: str, pooler_cls: Type['CallPooler']):
        """
        Register an external pooler class at runtime.

        Use this for registering poolers from external packages,
        GitHub repositories, or dynamically loaded modules.

        Args:
            name: Name to register the pooler under.
            pooler_cls: The pooler class to register. Must inherit from CallPooler.

        Example:
            from external_lib import AdvancedPooler
            PoolerRegistry.register_external("advanced", AdvancedPooler)

            # Now usable via config
            pooler = PoolerRegistry.get("advanced", n_features=11)
        """
        cls._poolers[name] = pooler_cls

    @classmethod
    def available(cls) -> list[str]:
        """
        List all available pooler names.

        Returns:
            List of registered pooler names.
        """
        return list(cls._poolers.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a pooler is registered.

        Args:
            name: Name to check.

        Returns:
            True if a pooler is registered under this name.
        """
        return name in cls._poolers

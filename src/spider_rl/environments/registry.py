from .standard import SpiderEnv

ENV_VARIANTS = {
    "standard": SpiderEnv,
}


def get_env_class(name: str):
    key = name.lower()
    if key not in ENV_VARIANTS:
        valid = ", ".join(sorted(ENV_VARIANTS))
        raise ValueError(f"Entorno '{name}' desconocido. Opciones: {valid}")
    return ENV_VARIANTS[key]


__all__ = ["SpiderEnv", "get_env_class", "ENV_VARIANTS"]

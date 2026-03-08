#!/usr/bin/env python3
"""
Apply step-attribute patch to OpenEnv web_interface.py so number inputs
use step=0.01 (not default 1), allowing float values like lr_scale=0.02 and momentum_coef=0.9.
Idempotent: safe to run multiple times.
"""
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        # Discover path from current Python's site-packages
        import openenv.core.env_server.web_interface as m

        path = Path(m.__file__).resolve()
    else:
        path = Path(sys.argv[1]).resolve()

    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    text = path.read_text()

    # Already patched?
    if "_get_step_for_number_field" in text and '"step": _get_step_for_number_field(field_info)' in text:
        print("Already patched:", path)
        return

    # 1) Insert helper before _extract_action_fields
    old1 = 'def _extract_action_fields(action_cls: Type[Action]) -> List[Dict[str, Any]]:'
    new1 = '''def _get_step_for_number_field(field_info: Dict[str, Any]) -> float | None:
    """Step for number inputs; avoids HTML5 default of 1 which restricts float ranges."""
    if field_info.get("type") == "integer":
        return 1
    if field_info.get("type") == "number":
        return field_info.get("multipleOf") if field_info.get("multipleOf") is not None else 0.01
    return None


def _extract_action_fields(action_cls: Type[Action]) -> List[Dict[str, Any]]:'''
    if new1 not in text and old1 in text:
        text = text.replace(old1, new1, 1)

    # 2) Add "step" to action_fields dict
    old2 = '                "help_text": _generate_help_text(field_name, field_info),\n            }\n        )'
    new2 = '                "help_text": _generate_help_text(field_name, field_info),\n                "step": _get_step_for_number_field(field_info),\n            }\n        )'
    if new2 not in text and old2 in text:
        text = text.replace(old2, new2, 1)

    # 3) Add step_value and step attribute in _generate_single_field
    old3a = "    max_value = field.get(\"max_value\")\n    default_value = field.get(\"default_value\")"
    new3a = "    max_value = field.get(\"max_value\")\n    step_value = field.get(\"step\")\n    default_value = field.get(\"default_value\")"
    if new3a not in text and old3a in text:
        text = text.replace(old3a, new3a, 1)

    old3b = "    if default_value is not None:\n        input_attrs.append(f'value=\"{default_value}\"')\n\n    attrs_str = \" \".join(input_attrs)"
    new3b = "    if default_value is not None:\n        input_attrs.append(f'value=\"{default_value}\"')\n    if step_value is not None:\n        input_attrs.append(f'step=\"{step_value}\"')\n\n    attrs_str = \" \".join(input_attrs)"
    if new3b not in text and old3b in text:
        text = text.replace(old3b, new3b, 1)

    path.write_text(text)
    print("Patched:", path)


if __name__ == "__main__":
    main()

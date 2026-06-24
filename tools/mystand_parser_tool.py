"""Generated Xiaoban tool bridge for MyStand Parser Tools."""

from __future__ import annotations

import os
import sys

_PARSER_SRC = os.environ.get('MYSTAND_PARSER_PYTHONPATH', '/opt/mystand-parser-tools/src')
if _PARSER_SRC and _PARSER_SRC not in sys.path:
    sys.path.insert(0, _PARSER_SRC)

from mystand_parser_tools.xiaoban import (
    MYSTAND_PARSE_SCHEMA,
    check_mystand_parser,
    mystand_parse_tool_handler,
)
from tools.registry import registry

registry.register(
    name='mystand_parse',
    toolset='mystand_parser',
    schema=MYSTAND_PARSE_SCHEMA,
    handler=mystand_parse_tool_handler,
    check_fn=check_mystand_parser,
    requires_env=[],
    is_async=False,
    description='Parse files and URLs with MyStand Parser Tools',
    emoji='\U0001f4c4',
)

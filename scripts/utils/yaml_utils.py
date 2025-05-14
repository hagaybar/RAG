# scripts/utils/yaml_utils.py
import yaml

class SmartQuotedStringDumper(yaml.SafeDumper):
    """
    Dumper that quotes string values but leaves keys and numbers unquoted.
    """

    def represent_str(self, data):
        # Check if we're inside a key context by inspecting the parent node
        if self.context_stack and isinstance(self.context_stack[-1], yaml.MappingNode):
            is_key = (self.context_stack[-1].value.index(self.cur_node) % 2 == 0)
            if is_key:
                return super().represent_str(data)  # leave keys unquoted if possible
        return self.represent_scalar('tag:yaml.org,2002:str', data, style='"')

SmartQuotedStringDumper.add_representer(str, SmartQuotedStringDumper.represent_str)

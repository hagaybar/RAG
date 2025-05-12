# scripts/utils/yaml_utils.py
import yaml

class QuotedStringDumper(yaml.SafeDumper):
    """
    Custom YAML dumper that forces all string values to be enclosed in double quotes.
    See full documentation inside.
    """
    @staticmethod
    def represent_str(dumper, data):
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')

QuotedStringDumper.add_representer(str, QuotedStringDumper.represent_str)

# Author: Yuya HAGA


def parse_envi_header(lines: list) -> dict[str, str]:
    """
    Parses ENVI file content into a structured dictionary
    This code was written with Github Copilot
    """
    envi_data = {}
    in_block_key = None

    for line_org in lines:
        line = line_org.strip()

        # Skip empty lines
        if not line:
            continue

        # Handle multiline blocks
        if in_block_key:
            if line.endswith("}"):
                # Closing multiline
                envi_data[in_block_key] += line[:-1].strip()
                in_block_key = None
            else:
                # Continue multiline
                envi_data[in_block_key] += line
            continue

        # Key-value pair parsing
        if "=" in line:
            key, value = map(str.strip, line.split("=", 1))
            key = key.lower().replace(" ", "_")  # Normalize key format

            # Handle block values
            if value.startswith("{"):
                # Handle multiline block
                in_block_key = key
                # Remove opening '{'
                value = value[1:].strip()
                if value.endswith("}"):
                    # Single-line block
                    envi_data[key] = value[:-1]
                    in_block_key = None
                else:
                    # Start multiline block
                    envi_data[key] = value
            else:
                # Single-line value
                envi_data[key] = value
    return envi_data

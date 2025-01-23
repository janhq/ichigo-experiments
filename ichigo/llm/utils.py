def convert_ids_to_tokens(id_list):
    """
    Convert a list of IDs to a compressed sound token string.

    Args:
        id_list (list): List of sound IDs

    Returns:
        str: Formatted string with sound tokens and duration
    """
    if not id_list:
        return "<|sound_start|><|sound_end|>"

    result = ["<|sound_start|>"]
    i = 0

    while i < len(id_list):
        current_id = id_list[i]
        count = 1

        # Count consecutive occurrences of the same ID
        while i + count < len(id_list) and id_list[i + count] == current_id:
            count += 1

        # Add duration token if count > 1
        if count > 1:
            result.append(f"<|duration_{str(count).zfill(2)}|>")

        # Add the sound token (each ID separately)
        result.append(f"<|sound_{str(current_id).zfill(4)}|>")

        # Move index forward
        i += count

    result.append("<|sound_end|>")
    return "".join(result)

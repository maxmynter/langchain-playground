def remove_unicode_null(text):
    cleaned_text = text.replace('\u0000', '')
    return cleaned_text
def find_longest_length(text_file_paths):
    """
    Find the longest review length in the entire training set. 
    :param text_file_paths: List, containing all the text file paths.
    Returns:
        max_len: Longest review length.
    """
    max_length = 0
    for path in text_file_paths:
        with open(path, 'r') as f:
            text = f.read()
            # Remove <br> tags.
            text = re.sub('<[^>]+>+', '', text)
            corpus = [
                word for word in text.split()
            ]
        if len(corpus) > max_length:
            max_length = len(corpus)
    return max_length
file_paths = []
file_paths.extend(glob.glob(os.path.join(
    dataset_dir, 'train', 'pos', '*.txt'
)))
file_paths.extend(glob.glob(os.path.join(
    dataset_dir, 'train', 'neg', '*.txt'
)))
longest_sentence_length = find_longest_length(file_paths)
print(f"Longest review length: {longest_sentence_length} words") 
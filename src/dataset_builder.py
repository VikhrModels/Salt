from datasets import load_dataset, concatenate_datasets


def load_datasets(datasets_config, split_name="train"):
    if split_name not in datasets_config:
        raise ValueError(f"Split '{split_name}' not found in config")

    dataset_list = datasets_config[split_name]
    if not dataset_list:
        raise ValueError(f"No datasets for split '{split_name}'")

    print(f"\nLoading {len(dataset_list)} dataset(s) for '{split_name}' split\n")

    loaded = []
    for ds_config in dataset_list:
        name = ds_config["name"]
        split = ds_config["split"]
        columns = ds_config.get("columns", None)

        print(f"  → {name} ({split})")

        try:
            if columns:
                ds = load_dataset(name, split=split, columns=columns)
            else:
                ds = load_dataset(name, split=split)
            print(f"    ✓ {len(ds):,} examples\n")
            loaded.append(ds)
        except Exception as e:
            raise RuntimeError(f"Failed to load '{name}': {str(e)}")

    combined = concatenate_datasets(loaded)
    print(f"Combined: {len(combined):,} examples\n")

    return combined


def filter_by_length(dataset, tokenizer, max_text_length, max_audio_tokens, num_proc=1):
    initial = len(dataset)

    print("─" * 60)
    print(f"Filtering dataset")
    print("─" * 60)
    print(f"  Text tokens:  ≤ {max_text_length}")
    print(f"  Audio tokens: ≤ {max_audio_tokens}")
    print(f"  Initial size: {initial:,} examples")
    print()

    def is_valid(example):
        text_tokens = tokenizer(example.get("text", ""), add_special_tokens=False)[
            "input_ids"
        ]
        audio_tokens = example.get("audio_tokens", [])
        return (
            len(text_tokens) <= max_text_length
            and len(audio_tokens) <= max_audio_tokens
        )

    filtered = dataset.filter(is_valid, num_proc=num_proc)
    final = len(filtered)
    removed = initial - final
    percent = (removed / initial * 100) if initial > 0 else 0

    print(f"  Final size:   {final:,} examples")
    print(f"  Removed:      {removed:,} ({percent:.1f}%)")
    print("─" * 60 + "\n")

    return filtered

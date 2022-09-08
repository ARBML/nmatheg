# https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/run_qa_no_trainer.py
import re
import copy 

def overflow_to_sample_mapping(tokens, offsets, idx, max_len = 384, doc_stride = 128):
    fixed_tokens = []
    fixed_offsets = []
    sep_index = tokens.index(-100)
    question = tokens[:sep_index]
    context = tokens[sep_index+1:]
    q_offsets = offsets[:sep_index]
    c_offsets = offsets[sep_index+1:]
    q_len = len(question)
    c_len = len(context)
    st_idx = 0 
    samplings = []
    sequences = []

    # print('length for the question ', len(question))
    # print('length for the context ', len(context))
    while True:
        ed_idx = st_idx+max_len-q_len-1
        pad_re = max_len - len(question+[0] + context[st_idx:ed_idx])
        # print('pad_re ', pad_re, ' st_idx ', st_idx, ' ed_idx ', ed_idx)
        curr_tokens = question+[0] + context[st_idx:ed_idx] + [0] * pad_re
        curr_offset = q_offsets+[(0,0)] + c_offsets[st_idx:ed_idx] + [(0,0)] * pad_re
        curr_seq = [0]*q_len+[None]+[1]*(max_len-q_len-1)

        assert len(curr_tokens) == len(curr_offset) == len(curr_seq) == max_len, f"curr_tokens: {len(curr_tokens)}, curr_seq: {len(curr_seq)}"
        fixed_tokens.append(curr_tokens[:max_len])
        fixed_offsets.append(curr_offset[:max_len])
        samplings.append(idx)
        sequences.append(curr_seq)

        st_idx += doc_stride
        if pad_re > 0:
          break
    for i in range(len(fixed_tokens)):
      assert len(fixed_tokens[i]) == len(fixed_offsets[i])  
    return fixed_tokens, fixed_offsets, samplings, sequences

def prepare_features(examples, tokenizer, data_config, model_name = 'bert', max_len = 128):
    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    if 'bert' in model_name:
        tokenized_examples = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=384,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples["offset_mapping"]
        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]
        
        return tokenized_examples
        
    else:
        question_col, context_col = data_config['text'].split(",")
        tokenized_examples = copy.deepcopy(examples)
        input_ids = []
        offset_mapping = []
        sequence_ids = []
        sample_mapping = []

        for i, (question, context) in enumerate(zip(examples[question_col], examples[context_col])):
            offsets = []
            tokens = []
            sequences = []
            question_context = question + " <sp> "+context
            st = 0
            for word in question_context.split(" "):
                if word == '':
                  continue
                
                if word == ' ':
                  continue 

                if word == "<sp>":
                    offsets.append((0, 0))
                    tokens.append(-100) #TODO fix this one
                    st = 0 
                else:    
                    token_ids = tokenizer._encode_word(word)
                    token_ids = [token_id for token_id in token_ids if token_id != tokenizer.sow_idx]
                    token_strs = tokenizer._tokenize_word(word, remove_sow=True)
                    assert len(token_ids) == len(token_strs)
                    for j, token_id in enumerate(token_ids):
                        token_str = token_strs[j]
                        tokens.append(token_id)
                        offsets.append((st, st+len(token_str)))
                        st += len(token_str)
                    st += 1 # for space
            tokens, offsets, samplings, sequences = overflow_to_sample_mapping(tokens, offsets, i, max_len = max_len)
            sample_mapping += samplings
            input_ids += tokens
            offset_mapping += offsets
            sequence_ids += sequences
     
        tokenized_examples = {'input_ids':input_ids, 'sequence_ids':sequence_ids, 'offset_mapping': offset_mapping, 'overflow_to_sample_mapping': sample_mapping}
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            # cls_index = input_ids.index(tokenizer.cls_token_id)
            cls_index = 0 

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples['sequence_ids'][i]

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = tokenized_examples["overflow_to_sample_mapping"][i]

            answers = examples["answers"][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples['sequence_ids'][i]
            context_index = 1

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
   
        return tokenized_examples
import logging
from typing import Dict, Tuple, List

from overrides import overrides
import torch
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import util
from allennlp.training.metrics import Metric
from allennlp.nn.beam_search import BeamSearch


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("copynet")
class CopyNet(Model):
    """
    This is an implementation of `CopyNet <https://arxiv.org/pdf/1603.06393>`_.
    CopyNet is a sequence-to-sequence encoder-decoder model with a copying mechanism
    that can copy tokens from the source sentence into the target sentence instead of
    generating all target tokens only from the target vocabulary.

    It is very similar to a typical seq2seq model used in neural machine translation
    tasks, for example, except that in addition to providing a "generation" score at each timestep
    for the tokens in the target vocabulary, it also provides a "copy" score for each
    token that appears in the source sentence. In other words, you can think of CopyNet
    as a seq2seq model with a dynamic target vocabulary that changes based on the tokens
    in the source sentence, allowing it to predict tokens that are out-of-vocabulary (OOV)
    with respect to the actual target vocab.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies.
    source_embedder : ``TextFieldEmbedder``, required
        Embedder for source side sequences
    encoder : ``Seq2SeqEncoder``, required
        The encoder of the "encoder/decoder" model
    attention : ``Attention``, required
        This is used to get a dynamic summary of encoder outputs at each timestep
        when producing the "generation" scores for the target vocab.
    beam_size : ``int``, required
        Beam width to use for beam search prediction.
    max_decoding_steps : ``int``, required
        Maximum sequence length of target predictions.
    target_embedding_dim : ``int``, optional (default = 30)
        The size of the embeddings for the target vocabulary.
    copy_token : ``str``, optional (default = '@COPY@')
        The token used to indicate that a target token was copied from the source.
        If this token is not already in your target vocabulary, it will be added.
    source_namespace : ``str``, optional (default = 'source_tokens')
        The namespace for the source vocabulary.
    target_namespace : ``str``, optional (default = 'target_tokens')
        The namespace for the target vocabulary.
    metrics : ``List[Metric]``, optional (default = None)
        List of metrics to track.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 attention: Attention,
                 beam_size: int,
                 max_decoding_steps: int,
                 target_embedding_dim: int = 30,
                 copy_token: str = "@COPY@",
                 source_namespace: str = "source_tokens",
                 target_namespace: str = "target_tokens",
                 metrics: List[Metric] = None) -> None:
        super(CopyNet, self).__init__(vocab)
        self._metrics = metrics
        self._source_namespace = source_namespace
        self._target_namespace = target_namespace
        self._src_start_index = self.vocab.get_token_index(START_SYMBOL, self._source_namespace)
        self._src_end_index = self.vocab.get_token_index(END_SYMBOL, self._source_namespace)
        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)
        self._oov_index = self.vocab.get_token_index(self.vocab._oov_token, self._target_namespace)  # pylint: disable=protected-access
        self._copy_index = self.vocab.get_token_index(copy_token, self._target_namespace)
        if self._copy_index == self._oov_index:
            raise ConfigurationError(f"Special copy token {copy_token} missing from target vocab namespace. "
                                     f"You can ensure this token is added to the target namespace with the "
                                     f"vocabulary parameter 'tokens_to_add'.")

        self._target_vocab_size = self.vocab.get_vocab_size(self._target_namespace)

        # Encoding modules.
        self._source_embedder = source_embedder
        self._encoder = encoder

        # Decoder output dim needs to be the same as the encoder output dim since we initialize the
        # hidden state of the decoder with the final hidden state of the encoder.
        # We arbitrarily set the decoder's input dimension to be the same as the output dimension.
        self.encoder_output_dim = self._encoder.get_output_dim()
        self.decoder_output_dim = self.encoder_output_dim
        self.decoder_input_dim = self.decoder_output_dim

        target_vocab_size = self.vocab.get_vocab_size(self._target_namespace)

        # The decoder input will be a function of the embedding of the previous predicted token,
        # an attended encoder hidden state called the "attentive read", and another
        # weighted sum of the encoder hidden state called the "selective read".
        # While the weights for the attentive read are calculated by an `Attention` module,
        # the weights for the selective read are simply the predicted probabilities
        # corresponding to each token in the source sentence from the previous timestep.
        self._target_embedder = Embedding(target_vocab_size, target_embedding_dim)
        self._attention = attention
        self._input_projection_layer = Linear(
                target_embedding_dim + self.encoder_output_dim * 2,
                self.decoder_input_dim)

        # We then run the projected decoder input through an LSTM cell to produce
        # the next hidden state.
        self._decoder_cell = LSTMCell(self.decoder_input_dim, self.decoder_output_dim)

        # We create a "generation" score for each token in the target vocab
        # with a linear projection of the decoder hidden state.
        self._output_generation_layer = Linear(self.decoder_output_dim, target_vocab_size)

        # We create a "copying" score for each source token by applying a non-linearity
        # (tanh) to a linear projection of the encoded hidden state for that token,
        # and then taking the dot product of the result with the decoder hidden state.
        self._output_copying_layer = Linear(self.encoder_output_dim, self.decoder_output_dim)

        # At prediction time, we'll use a beam search to find the best target sequence.
        self._beam_search = BeamSearch(self._end_index, max_steps=max_decoding_steps, beam_size=beam_size)

    @overrides
    def forward(self,  # type: ignore
                source_tokens: Dict[str, torch.LongTensor],
                source_duplicates: torch.Tensor,
                target_pointers: torch.Tensor,
                target_tokens: Dict[str, torch.LongTensor] = None,
                copy_indicators: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Make foward pass with decoder logic for producing the entire target sequence.

        Parameters
        ----------
        source_tokens : ``Dict[str, torch.LongTensor]``, required
            The output of `TextField.as_array()` applied on the source `TextField`. This will be
            passed through a `TextFieldEmbedder` and then through an encoder.
        source_duplicates : ``torch.Tensor``, required
            Tensor containing indicators of which source tokens match each other.
            Has shape: `(batch_size, trimmed_source_length, trimmed_source_length)`.
        target_pointers : ``torch.Tensor``, required
            Tensor containing vocab index of each source token with respect to the
            target vocab namespace. Shape: `(batch_size, trimmed_source_length)`.
        target_tokens : ``Dict[str, torch.LongTensor]``, optional (default = None)
            Output of `Textfield.as_array()` applied on target `TextField`. We assume that the
            target tokens are also represented as a `TextField`.
        copy_indicators : ``Dict[str, torch.LongTensor]``, optional (default = None)
            A sparse tensor of shape `(batch_size, target_sequence_length, source_sentence_length - 2)` that
            indicates which tokens in the source sentence match each token in the target sequence.
            The last dimension is `source_sentence_length - 2` because we exclude the
            START_SYMBOL and END_SYMBOL in the source sentence (the source sentence is guaranteed
            to contain the START_SYMBOL and END_SYMBOL).

        Returns
        -------
        Dict[str, torch.Tensor]
        """
        state = self._init_encoded_state(source_tokens, source_duplicates, target_pointers)

        if target_tokens:
            output_dict = self._forward_loop(target_tokens, copy_indicators, state)
        else:
            output_dict = {}

        if not self.training:
            predictions = self._forward_beam_search(state)
            output_dict.update(predictions)
            if self._metrics:
                for metric in self._metrics:
                    metric(source_tokens, target_tokens, copy_indicators, predictions)

        return output_dict

    def _init_encoded_state(self,
                            source_tokens: Dict[str, torch.Tensor],
                            source_duplicates: torch.Tensor,
                            target_pointers: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Initialize the encoded state to be passed to the first decoding time step.
        """
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input = self._source_embedder(source_tokens)

        batch_size, _, _ = embedded_input.size()

        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)

        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self._encoder(embedded_input, source_mask)

        # shape: (batch_size, encoder_output_dim)
        final_encoder_output = util.get_final_encoder_states(
                encoder_outputs,
                source_mask,
                self._encoder.is_bidirectional())

        # Initialize the decoder hidden state with the final output of the encoder.
        # shape: (batch_size, decoder_output_dim)
        decoder_hidden = final_encoder_output

        # shape: (batch_size, decoder_output_dim)
        decoder_context = encoder_outputs.new_zeros(batch_size, self.decoder_output_dim)

        state = {
                "source_mask": source_mask,
                "encoder_outputs": encoder_outputs,
                "decoder_hidden": decoder_hidden,
                "decoder_context": decoder_context,
                "source_duplicates": source_duplicates,
                "target_pointers": target_pointers,
        }

        return state

    def _decoder_step(self,
                      last_predictions: torch.Tensor,
                      selective_weights: torch.Tensor,
                      state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs_mask = state["source_mask"].float()

        # shape: (group_size, target_embedding_dim)
        embedded_input = self._target_embedder(last_predictions)

        # shape: (batch_size, max_input_sequence_length)
        attentive_weights = self._attention(
                state["decoder_hidden"], state["encoder_outputs"], encoder_outputs_mask)

        # shape: (batch_size, encoder_output_dim)
        attentive_read = util.weighted_sum(state["encoder_outputs"], attentive_weights)

        # shape: (batch_size, encoder_output_dim)
        selective_read = util.weighted_sum(state["encoder_outputs"][:, 1:-1], selective_weights)

        # shape: (group_size, target_embedding_dim + encoder_output_dim * 2)
        decoder_input = torch.cat((embedded_input, attentive_read, selective_read), -1)

        # shape: (group_size, decoder_input_dim)
        projected_decoder_input = self._input_projection_layer(decoder_input)

        state["decoder_hidden"], state["decoder_context"] = self._decoder_cell(
                projected_decoder_input,
                (state["decoder_hidden"], state["decoder_context"]))

        return state

    def _get_generation_scores(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._output_generation_layer(state["decoder_hidden"])

    def _get_copy_scores(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        # shape: (batch_size, max_input_sequence_length - 2, encoder_output_dim)
        trimmed_encoder_outputs = state["encoder_outputs"][:, 1:-1]

        # shape: (batch_size, max_input_sequence_length - 2, decoder_output_dim)
        copy_projection = self._output_copying_layer(trimmed_encoder_outputs)

        # shape: (batch_size, max_input_sequence_length - 2, decoder_output_dim)
        copy_projection = torch.tanh(copy_projection)

        # shape: (batch_size, max_input_sequence_length - 2)
        copy_scores = copy_projection.bmm(state["decoder_hidden"].unsqueeze(-1)).squeeze(-1)

        return copy_scores

    def _get_ll_contrib(self,
                        generation_scores: torch.Tensor,
                        copy_scores: torch.Tensor,
                        target_tokens: torch.Tensor,
                        copy_indicators: torch.Tensor,
                        copy_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the log-likelihood contribution from a single timestep.

        Parameters
        ----------
        generation_scores : ``torch.Tensor``
            Shape: `(batch_size, target_vocab_size)`
        copy_scores : ``torch.Tensor``
            Shape: `(batch_size, trimmed_source_length)`
        target_tokens : ``torch.Tensor``
            Shape: `(batch_size,)`
        copy_indicators : ``torch.Tensor``
            Shape: `(batch_size, trimmed_source_length)`
        copy_mask : ``torch.Tensor``
            Shape: `(batch_size, trimmed_source_length)`

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Shape: `(batch_size,), (batch_size, max_input_sequence_length)`
        """
        _, target_size = generation_scores.size()

        # The point of this mask is to just mask out all source token scores
        # that just represent padding. We apply the mask to the concatenation
        # of the generation scores and the copy scores to normalize the scores
        # correctly during the softmax.
        # shape: (batch_size, target_vocab_size + trimmed_source_length)
        mask = torch.cat((generation_scores.new_full(generation_scores.size(), 1.0), copy_mask), dim=-1)

        # shape: (batch_size, target_vocab_size + trimmed_source_length)
        all_scores = torch.cat((generation_scores, copy_scores), dim=-1)

        # Normalize generation and copy scores.
        # shape: (batch_size, target_vocab_size + trimmed_source_length)
        probs = util.masked_softmax(all_scores, mask)

        # Calculate the probability (normalized copy score) for each token in the source sentence
        # that matches the current target token. We end up summing the scores
        # for each occurence of a matching token to get the actual score, but we also
        # need the un-summed probabilities to create the selective read state
        # during the next time step.
        # shape: (batch_size, trimmed_source_length)
        selective_weights = probs[:, target_size:] * copy_indicators.float()

        # This mask ensures that item in the batch has a non-zero generation score for this timestep
        # only when the gold target token is not OOV or there are no matching tokens
        # in the source sentence.
        # shape: (batch_size,)
        gen_mask = ((target_tokens != self._oov_index) | (copy_indicators.sum(-1) == 0)).float()

        # Now we get the generation score for the gold target token.
        # shape: (batch_size,)
        step_prob = probs.gather(1, target_tokens.unsqueeze(1)).squeeze(-1) * gen_mask

        # ... and add the copy score.
        # shape: (batch_size,)
        step_prob = step_prob + selective_weights.sum(-1)

        # shape: (batch_size,)
        step_log_likelihood = step_prob.log()

        return step_log_likelihood, selective_weights

    def _forward_loop(self,
                      target_tokens: Dict[str, torch.LongTensor],
                      copy_indicators: torch.Tensor,
                      state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculate the loss against gold targets.
        """
        # shape: (batch_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        batch_size, target_sequence_length = target_tokens["tokens"].size()

        # The last input from the target is either padding or the end symbol.
        # Either way, we don't have to process it.
        num_decoding_steps = target_sequence_length - 1

        # We initialize the target predictions with the start token.
        # shape: (batch_size,)
        input_choices = source_mask.new_full((batch_size,), fill_value=self._start_index)

        # We use this to fill in the copy index when the previous input was copied.
        # shape: (batch_size,)
        copy_input_choices = source_mask.new_full((batch_size,), fill_value=self._copy_index)

        # shape: (batch_size, trimmed_source_length)
        copy_mask = source_mask[:, 1:-1].float()

        # We need to keep track of the probabilities assigned to tokens in the source
        # sentence that were copied during the previous timestep, since we use
        # those probabilities as weights when calculating the "selective read".
        # shape: (batch_size, trimmed_source_length)
        selective_weights = state["decoder_hidden"].new_zeros(copy_mask.size())

        step_log_likelihoods = []
        for timestep in range(num_decoding_steps):
            # shape: (batch_size,)
            input_choices = target_tokens["tokens"][:, timestep]

            # If the previous target token was copied, we use the special copy token.
            # But the end target token will always be THE end token, so we know
            # it was not copied.
            if timestep < num_decoding_steps - 1:
                # Get mask tensor indicating which instances were copied.
                # shape: (batch_size,)
                copied = (copy_indicators[:, timestep, :].sum(-1) > 0).long()

                # shape: (batch_size,)
                input_choices = input_choices * (1 - copied) + copy_input_choices * copied

            # Update the decoder state by taking a step through the RNN.
            state = self._decoder_step(input_choices, selective_weights, state)

            # Get generation scores for each token in the target vocab.
            # shape: (batch_size, target_vocab_size)
            generation_scores = self._get_generation_scores(state)

            # Get copy scores for each token in the source sentence, excluding the start
            # and end tokens.
            # shape: (batch_size, max_input_sequence_length - 2)
            copy_scores = self._get_copy_scores(state)

            # shape: (batch_size,)
            step_target_tokens = target_tokens["tokens"][:, timestep + 1]

            # shape: (batch_size, max_input_sequence_length - 2)
            step_copy_indicators = copy_indicators[:, timestep + 1]

            step_log_likelihood, selective_weights = self._get_ll_contrib(
                    generation_scores,
                    copy_scores,
                    step_target_tokens,
                    step_copy_indicators,
                    copy_mask)
            step_log_likelihoods.append(step_log_likelihood.unsqueeze(1))

        # Gather step log-likelihoods.
        # shape: (batch_size, num_decoding_steps = target_sequence_length - 1)
        log_likelihoods = torch.cat(step_log_likelihoods, 1)

        # Get target mask to exclude likelihood contributions from timesteps after
        # the END token.
        # shape: (batch_size, target_sequence_length)
        target_mask = util.get_text_field_mask(target_tokens)

        # The first timestep is just the START token, which is not included in the likelihoods.
        # shape: (batch_size, num_decoding_steps)
        target_mask = target_mask[:, 1:].float()

        # Sum of step log-likelihoods.
        # shape: (batch_size,)
        log_likelihood = (log_likelihoods * target_mask).sum(dim=-1)

        # The loss is the negative log-likelihood, averaged over the batch.
        loss = - log_likelihood.sum() / batch_size

        return {"loss": loss}

    def _get_selective_weights(self,
                               last_predictions: torch.LongTensor,
                               copied: torch.LongTensor,
                               state: Dict[str, torch.Tensor]) -> torch.Tensor:
        # shape: (group_size, trimmed_source_length)
        copy_probs = state["copy_probs"]

        # shape: (group_size, trimmed_source_length,  trimmed_source_length)
        source_duplicates = state["source_duplicates"]

        # Adjust predictions relative to start of source tokens. This makes sense
        # because predictions for copied tokens are given by the index of the copied
        # token in the source sentence, offset by the size of the target vocabulary.
        # shape: (group_size,)
        adjusted_predictions = last_predictions - self._target_vocab_size

        # The adjusted indices for items that were not copied will be negative numbers,
        # and therefore invalid. So we zero them out.
        adjusted_predictions = adjusted_predictions * copied

        # Expand adjusted_predictions to match source_duplicates shape.
        # shape: (group_size, trimmed_source_length, trimmed_source_length)
        adjusted_predictions = adjusted_predictions.unsqueeze(-1)\
            .unsqueeze(-1)\
            .expand(source_duplicates.size())

        # The mask will contain indicators for source tokens that were copied
        # during the last timestep.
        # shape: (group_size, trimmed_source_length)
        mask = source_duplicates.gather(-1, adjusted_predictions)[:, :, 0]

        # Since we zero'd-out indices for predictions that were not copied,
        # we need to zero out all entries of this mask corresponding to those predictions.
        mask = mask * copied.unsqueeze(-1).expand(mask.size())

        return copy_probs * mask.float()

    def take_search_step(self,
                         last_predictions: torch.Tensor,
                         state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Take step during beam search.

        This function is what gets passed to the `BeamSearch.search` method. It takes
        predictions from the last timestep and the current state and outputs
        the log probabilities assigned to tokens for the next timestep, as well as the updated
        state.

        Since we are predicting tokens out of the extended vocab (target vocab + all unique
        tokens from the source sentence), this is a little more complicated that just
        making a forward pass through the model. The output log probs will have
        shape `(group_size, target_vocab_size + trimmed_source_length)` so that each
        token in the target vocab and source sentence are assigned a probability.

        Note that copy scores are assigned to each source token based on their position, not unique value.
        So if a token appears more than once in the source sentence, it will have more than one score.
        Further, if a source token is also part of the target vocab, its final score
        will be the sum of the generation and copy scores. Therefore, in order to
        get the score for all tokens in the extended vocab at this step,
        we have to combine copy scores for re-occuring source tokens and potentially
        add them to the generation scores for the matching token in the target vocab, if
        there is one.

        So we can break down the final log probs output as the concatenation of two
        matrices, A: `(group_size, target_vocab_size)`, and B: `(group_size, trimmed_source_length)`.
        Matrix A contains the sum of the generation score and copy scores (possibly 0)
        for each target token. Matrix B contains left-over copy scores for source tokens
        that do NOT appear in the target vocab, with zeros everywhere else. But since
        a source token may appear more than once in the source sentence, we also have to
        sum the scores for each appearance of each unique source token. So matrix B
        actually only has non-zero values at the first occurence of each source token
        that is not in the target vocab.

        Parameters
        ----------
        last_predictions : ``torch.Tensor``
            Shape: `(group_size,)`

        state : ``Dict[str, torch.Tensor]``
            Contains all state tensors necessary to produce generation and copy scores
            for next step.

        Notes
        -----
        `group_size` != `batch_size`. In fact, `group_size` = `batch_size * beam_size`.
        """
        group_size, source_length = state["source_mask"].size()
        trimmed_source_length = source_length - 2

        # Get mask tensor indicating which predictions were copied.
        # shape: (group_size,)
        copied = (last_predictions >= self._target_vocab_size).long()

        # Set selective_weights to probs assigned to copied source tokens from
        # last timestep.
        # (group_size,  trimmed_source_length)
        selective_weights = self._get_selective_weights(last_predictions, copied, state)

        # We use this to fill in the copy index when the previous input was copied.
        # shape: (group_size,)
        copy_input_choices = copied.new_full((group_size,), fill_value=self._copy_index)

        # So if the last prediction was in the target vocab or OOV but not copied,
        # we use that as input, otherwise we use the COPY token.
        # shape: (group_size,)
        input_choices = last_predictions * (1 - copied) + copy_input_choices * copied

        # Update the decoder state by taking a step through the RNN.
        state = self._decoder_step(input_choices, selective_weights, state)

        # Get the un-normalized generation scores for each token in the target vocab.
        # shape: (group_size, target_vocab_size)
        generation_scores = self._get_generation_scores(state)

        # Get the un-normalized copy scores for each token in the source sentence,
        # excluding the start and end tokens.
        # shape: (group_size, trimmed_source_length)
        copy_scores = self._get_copy_scores(state)

        # Concat un-normalized generation and copy scores.
        # shape: (batch_size, target_vocab_size + trimmed_source_length)
        all_scores = torch.cat((generation_scores, copy_scores), dim=-1)

        # shape: (group_size, trimmed_source_length)
        copy_mask = state["source_mask"][:, 1:-1].float()

        # shape: (batch_size, target_vocab_size + trimmed_source_length)
        mask = torch.cat((generation_scores.new_full(generation_scores.size(), 1.0), copy_mask), dim=-1)

        # Normalize generation and copy scores.
        # shape: (batch_size, target_vocab_size + trimmed_source_length)
        probs = util.masked_softmax(all_scores, mask)

        # shape: (group_size, target_vocab_size), (group_size, trimmed_source_length)
        generation_probs, copy_probs = probs.split([self._target_vocab_size, trimmed_source_length], dim=-1)

        # Update copy_probs needed for getting the `selective_weights` at the next timestep.
        state["copy_probs"] = copy_probs

        # We now have normalized generation and copy scores, but to produce the final
        # score for each token in the extended vocab, we have to go through and add
        # the copy scores to the generation scores of matching target tokens, and sum
        # the copy scores of duplicate source tokens.
        # shape: [(batch_size, *)]
        modified_probs_list: List[torch.Tensor] = [generation_probs]
        for i in range(trimmed_source_length):
            # shape: (group_size,)
            copy_probs_slice = copy_probs[:, i]

            # `target_pointers` is a matrix of shape (group_size, trimmed_source_length)
            # where element (i, j) is the vocab index of the target token that matches the jth
            # source token in the ith group, if there is one, or the index of the OOV symbol otherwise.
            # We'll use this to add copy scores to corresponding generation scores.
            # shape: (group_size,)
            target_pointers_slice = state["target_pointers"][:, i]

            # The OOV index in the target_pointers_slice indicates that the source
            # token is not in the target vocab, so we don't want to add that copy score
            # to the OOV token.
            copy_probs_to_add_mask = (target_pointers_slice != self._oov_index).float()
            copy_probs_to_add = copy_probs_slice * copy_probs_to_add_mask
            generation_probs.scatter_add_(
                    -1, target_pointers_slice.unsqueeze(-1), copy_probs_to_add.unsqueeze(-1))

            # We now zero-out copy scores that we added to the generation scores
            # above so that we don't double-count them.
            # shape: (group_size,)
            left_over_copy_probs = copy_probs_slice * (1.0 - copy_probs_to_add_mask)

            # Lastly, we have to combine copy scores for duplicate source tokens so that
            # we can find the overall most likely source token. So, if this is the first
            # occurence of this particular source token, we add the probs from all other
            # occurences, otherwise we zero it out since it was already accounted for.
            if i > 0:
                # Zero-out copy probs that we have already accounted for.
                # shape: (group_size, i)
                source_previous_occurences = state["source_duplicates"][:, :, 0:i]
                # shape: (group_size,)
                duplicate_mask = (source_previous_occurences.sum(dim=-1) > 0).float()
                left_over_copy_probs = left_over_copy_probs * duplicate_mask
            if i < (trimmed_source_length - 1):
                # Sum copy scores from future occurences of source token.
                # shape: (group_size, trimmed_source_length - i)
                source_future_occurences = state["source_duplicates"][:, :, (i+1):]
                # shape: (group_size, trimmed_source_length - i)
                future_copy_probs = copy_probs[:, (i+1):] * source_future_occurences
                # shape: (group_size,)
                summed_future_copy_probs = future_copy_probs.sum(dim=-1)
                left_over_copy_probs = left_over_copy_probs + summed_future_copy_probs

            modified_probs_list.append(left_over_copy_probs.unsqueeze(-1))

        # shape: (group_size, target_vocab_size + trimmed_source_length)
        modified_probs = torch.cat(modified_probs_list, dim=-1)

        return modified_probs, state

    def _forward_beam_search(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size, _ = state["source_mask"].size()

        # shape: (batch_size,)
        start_predictions = state["source_mask"].new_full((batch_size,), fill_value=self._start_index)

        # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, beam_size)
        all_top_k_predictions, log_probabilities = self._beam_search.search(
                start_predictions, state, self.take_search_step)

        output_dict = {
                "predicted_log_probs": log_probabilities,
                "predictions": all_top_k_predictions,
        }

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Finalize predictions.

        After a beam search, the predicted indices correspond to tokens in the target vocabulary
        OR tokens in source sentence. Here we gather the actual tokens corresponding to
        the indices.
        """
        # TODO
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if self._metrics and not self.training:
            for metric in self._metrics:
                all_metrics.update(metric.get_metric(reset=reset))
        return all_metrics

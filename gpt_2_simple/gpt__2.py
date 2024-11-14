import os
import shutil
import regex as re
import numpy as np

import json
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.core.protobuf import rewriter_config_pb2
from functools import lru_cache
import tqdm
import csv
import glob
import random
import codecs
import time



def start_tf_sess(threads=-1, server=None):
    """
    Returns a tf.Session w/ config
    """
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.graph_options.rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF
    if threads > 0:
        config.intra_op_parallelism_threads = threads
        config.inter_op_parallelism_threads = threads

    if server is not None:
        return tf.compat.v1.Session(target=server.target, config=config)

    return tf.compat.v1.Session(config=config)

class HParams():
    """
    Hyperparameters:

    Examples:
        Learning Rate: Determines the step size at each iteration while moving toward a minimum of the loss function.
        Batch Size: The number of training examples utilized in one iteration.
        Number of Epochs: The number of complete passes through the training dataset.
        Number of Layers: In deep learning, this can include the number of hidden layers in a neural network.
        Number of Units/Neurons: The number of neurons in each layer of a neural network.
        Dropout Rate: The fraction of neurons to drop during training to prevent overfitting.
        Activation Functions: Functions applied to each neuron’s output in a neural network (e.g., GELU, ReLU, sigmoid, tanh).
        Optimization Algorithms: Methods used to minimize the loss function (e.g., SGD, Adam, RMSprop).
    """
    def __init__(
            self, 
            n_vocab=50257,
            n_ctx=1024,
            n_embd=768,
            n_head=12,
            n_layer=12):
        """ 
        n_vocab (int, optional):\n
        n_ctx (int, optional): context length or maximum sequence length that the model can handle. (maximum number of tokens, words or subwords, the model can process at one time.\n
        n_embd (int, optional): embedding size. "The embedding sublayer converts the input tokens to vectors of dimension *n_embd"\n
        n_head (int, optional):\n
        n_layer (int, optional):\n
        
        Attributes: \n
        attribute1 (int): Description of attribute1. 
        attribute2 (int): Description of attribute2. 
        """
        self.n_vocab = n_vocab
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer

    def to_string(self):
        return f"HParams(n_vocab={self.n_vocab}, n_ctx={self.n_ctx}, n_embd={self.n_embd}, n_head={self.n_head}, n_layer={self.n_layer})"


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_pairs(word):
    """Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class Encoder:
    def __init__(self, encoder, bpe_merges, errors='replace'):
        self.encoder = encoder
        self.decoder = {v:k for k,v in self.encoder.items()}
        self.errors = errors # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v:k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def bpe(self, token):
        """
        Byte-Pair Encoding:

        A BPE tokenizer will break a string or word down into substrings or subwords.
        There are two main advantages to this, amonng many others:

        1) The tokenizer can break words into minimal components. Then it will merge
           these small components into statistically interesting ones. For example,
           "smaller" and "smallest" can become "small", "er", and "est". The tokenizer
           can go further. We would get "sm" and "all", for example. In any case, the
           words are broken down into subword tokens and smaller units of subword parts
           such as "sm" and "all" instead of simply "small".
        2) The chunks of strings classified as unknown, unk_token, using WordPiece level
           encoding, will proactically disappear.
        """
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text

def get_encoder(checkpoint_path):
    with open(os.path.join(checkpoint_path, 'encoder.json'), 'r') as f:
        encoder = json.load(f)
    with open(os.path.join(checkpoint_path, 'vocab.bpe'), 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    return Encoder(
        encoder=encoder,
        bpe_merges=bpe_merges,
    )

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(input=x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def expand_tile(value, size):
    """Add a new axis of given size."""
    # Converts the input value to a tensor.
    value = tf.convert_to_tensor(value=value, name='value')
    # Retrieves the number of dimensions of the tensor.
    ndims = value.shape.ndims
    # Adds a new dimension at the beginning of the tensor.
    expanded_value = tf.expand_dims(value, axis=0)
    # Duplicates (tiles) the tensor along the new dimension size times.
    return tf.tile(expanded_value, [size] + [1]*ndims)

def positions_for(tokens, past_length):
    # Retrieves the batch size, i.e., the number of sequences in the batch.
    batch_size = tf.shape(input=tokens)[0]
    # Retrieves the number of steps (or length) of each sequence.
    nsteps = tf.shape(input=tokens)[1]
    # Generates a range of position indices, starting from past_length.
    position_indices = past_length + tf.range(nsteps)
    return expand_tile(position_indices, batch_size)

def get_value(a):
    if hasattr(a, 'value'):
        return a.value
    return a

def gelu(x):
    """
    Activation Function: Approximation of GELU (Gaussian Error Linear Unit) function using hyperbolic tangent (tanh):
    """
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

def conv1d(x, scope, nf, *, w_init_stdev=0.02):
    """
    1D convolution operation

    x (): input tensor.
    scope (): variable scope for TensorFlow variables, ensuring that variable names are unique.
    nf (): number of filters (output features) for the convolution.
    w_init_stdev (float, optional): standard deviation for the weight initialization (default is 0.02).
    """
    with tf.compat.v1.variable_scope(scope):
        *start, nx = shape_list(x)
        # initializes the weight variable with shape [1, nx, nf] using a random normal initializer.
        w = tf.compat.v1.get_variable('w', [1, nx, nf], initializer=tf.compat.v1.random_normal_initializer(stddev=w_init_stdev))
        # initializes the bias variable with shape [nf] using a constant initializer, 0
        b = tf.compat.v1.get_variable('b', [nf], initializer=tf.compat.v1.constant_initializer(0))
        c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, start+[nf])
        return c

def mlp(x, scope, n_state, *, hparams: HParams):
    """
    Multi-Layer Perceptron:

    x (): input tensor.
    scope (): variable scope for TensorFlow variables to ensure unique naming.
    n_state (): number of units (neurons) in the hidden layer.
    hparams (): hyperparameters for the model, passed as an argument.
    """
    with tf.compat.v1.variable_scope(scope):
        num_features = get_value(x.shape[-1])
        hidden_layer_output = gelu(conv1d(x, 'c_fc', n_state))
        output_layer = conv1d(hidden_layer_output, 'c_proj', num_features)
        return output_layer
    
def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m//n])

def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a*b])


    
def multi_head_attention(x, scope, n_state, *, past, hparams):
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state % hparams.n_head == 0
    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(a=split_states(x, hparams.n_head), perm=[0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.transpose(a=x, perm=[0, 2, 1, 3]))
    
    def attention_mask(nd, ns, *, dtype):
        """1's in the lower triangle, counting from the lower right corner.

        Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
        """
        i = tf.range(nd)[:,None]
        j = tf.range(ns)
        m = i >= j - ns + nd
        return tf.cast(m, dtype)

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)
        return w
    
    def softmax(x, axis=-1):
        x = x - tf.reduce_max(input_tensor=x, axis=axis, keepdims=True)
        ex = tf.exp(x)
        return ex / tf.reduce_sum(input_tensor=ex, axis=axis, keepdims=True)

    def multihead_attn(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.math.rsqrt(tf.cast(get_value(v.shape[-1]), w.dtype))

        w = mask_attn_weights(w)
        w = softmax(w)
        a = tf.matmul(w, v)
        return a

    with tf.compat.v1.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state*3)
        q, k, v = map(split_heads, tf.split(c, 3, axis=2))
        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state)
        return a, present

def transformer_block(x_tensor, scope, *, past_states, hyperparameters: HParams):
    with tf.compat.v1.variable_scope(scope):
        num_features = get_value(x_tensor.shape[-1])
        attention_output, present_states = multi_head_attention(
            norm(x_tensor, 'ln_1'), 
            'attn', 
            num_features, 
            past=past_states, 
            hparams=hyperparameters)
        x_tensor = x_tensor + attention_output
        feed_forward_output = mlp(norm(x_tensor, 'ln_2'), 'mlp', num_features*4, hparams=hyperparameters)
        x_tensor = x_tensor + feed_forward_output
        return x_tensor, present_states
    
def norm(x_tensor, scope, *, axis=-1, epsilon=1e-5):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    with tf.compat.v1.variable_scope(scope):
        n_state = get_value(x_tensor.shape[-1])
        g = tf.compat.v1.get_variable('g', [n_state], initializer=tf.compat.v1.constant_initializer(1))
        b = tf.compat.v1.get_variable('b', [n_state], initializer=tf.compat.v1.constant_initializer(0))
        mean = tf.reduce_mean(input_tensor=x_tensor, axis=axis, keepdims=True)
        variance = tf.reduce_mean(input_tensor=tf.square(x_tensor-mean), axis=axis, keepdims=True)
        normalized_tensor = (x_tensor - mean) * tf.math.rsqrt(variance + epsilon)
        output_tensor = normalized_tensor*g + b
        return output_tensor

def build_transformer_model(hparams: HParams, X, past=None, scope='model', gpus=[], reuse=False):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        results = {}
        batch, sequence = shape_list(X)

        """
        wpe appears to stand for Position Embeddings. In Transformer models, 
        position embeddings are used to encode the position of tokens in a 
        sequence since Transformers lack inherent positional information.
        """
        # wpe: tf.Tensor of type float32
        wpe = tf.compat.v1.get_variable(
            'wpe', 
            [hparams.n_ctx, hparams.n_embd],
            initializer=tf.compat.v1.random_normal_initializer(stddev=0.01))
        """
        wte appears to stand for Word Token Embeddings. These embeddings map 
        tokens (words) to their corresponding vector representations.
        """
        # wte: tf.Tensor of type float32
        wte = tf.compat.v1.get_variable(
            'wte', 
            [hparams.n_vocab, hparams.n_embd],
            initializer=tf.compat.v1.random_normal_initializer(stddev=0.02))
        
        past_length = 0 if past is None else tf.shape(input=past)[-2]
        """
        token embeddings and position embeddings are combined to form the input embeddings.
        allows the model to consider both the identity and the position of each token when processing the input.
        """
        input_embeddings = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, past_length))

        # Transformer
        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        assert len(pasts) == hparams.n_layer
        gpu_stack = np.floor(hparams.n_layer/len(gpus)) if len(gpus) > 0 else 0
        d = 0
        for layer, past in enumerate(pasts):
            if gpu_stack < 1:
                input_embeddings, present = transformer_block(input_embeddings, 'h%d' % layer, past_states=past, hyperparameters=hparams)
                tf.compat.v1.add_to_collection('checkpoints', input_embeddings)
                presents.append(present)
            else:
                if layer != 0 and layer % gpu_stack == 0 and d+1 != len(gpus):
                    d += 1
                with tf.device(gpus[d]):
                    input_embeddings, present = transformer_block(input_embeddings, 'h%d' % layer, past_states=past, hyperparameters=hparams)
                    tf.compat.v1.add_to_collection('checkpoints', input_embeddings)
                    presents.append(present)
        results['present'] = tf.stack(presents, axis=1)
        input_embeddings = norm(input_embeddings, 'ln_f')

        # Language model loss.  Do tokens <n predict token n?
        h_flat = tf.reshape(input_embeddings, [batch*sequence, hparams.n_embd])
        logits = tf.matmul(h_flat, wte, transpose_b=True)
        logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])
        results['logits'] = logits
        return results
    
def past_shape(*, hparams: HParams, batch_size=None, sequence=None):
    shape = [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]
    print(f"past_shape: {shape}")
    return shape

def top_p_logits(logits, p):
    with tf.compat.v1.variable_scope('top_p_logits'):
        logits_sort = tf.sort(logits, direction='DESCENDING')
        probs_sort = tf.nn.softmax(logits_sort)
        probs_sums = tf.cumsum(probs_sort, axis=1, exclusive=True)
        logits_masked = tf.compat.v1.where(probs_sums < p, logits_sort, tf.ones_like(
            logits_sort)*1000)  # [batchsize, vocab]
        min_logits = tf.reduce_min(input_tensor=logits_masked, axis=1, keepdims=True)  # [batchsize, 1]
        return tf.compat.v1.where(
            logits < min_logits,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )
    
def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.compat.v1.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )
    return tf.cond(
        pred=tf.equal(k, 0),
        true_fn=lambda: logits,
        false_fn=lambda: _top_k(),
    )


def sample_sequence(*, hparams: HParams, length, start_token=None,
                    batch_size=None, context=None, temperature=1,
                    top_k=0, top_p=0.0):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = tf.fill([batch_size, 1], start_token)

    def step(hparams: HParams, tokens, past=None):
        lm_output = build_transformer_model(hparams=hparams, X=tokens,
                                past=past, reuse=tf.compat.v1.AUTO_REUSE)

        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        presents = lm_output['present']
        presents.set_shape(past_shape(
            hparams=hparams, batch_size=batch_size))
        return {
            'logits': logits,
            'presents': presents,
        }

    with tf.compat.v1.name_scope('sample_sequence'):
        # Don't feed the last context token -- leave that to the loop below
        # TODO: Would be slightly faster if we called step on the entire context,
        # rather than leaving the last token transformer calculation to the while loop.
        context_output = step(hparams, context[:, :-1])

        def body(past, prev, output):
            next_outputs = step(hparams, prev[:, tf.newaxis], past=past)
            logits = next_outputs['logits'][:, -1, :] / tf.cast(temperature, tf.float32)
            if top_p > 0.0:
                logits = top_p_logits(logits, p=top_p)
            else:
                logits = top_k_logits(logits, k=top_k)
            samples = tf.random.categorical(
                logits, num_samples=1, dtype=tf.int32)
            return [
                tf.concat([past, next_outputs['presents']], axis=-2),
                tf.squeeze(samples, axis=[1]),
                tf.concat([output, samples], axis=1),
            ]

        def cond(*args):
            return True

        """
        Map Structure: 
        Applies the "stop_gradient" to the result of the "while_loop"
        """
        _, _, tokens = tf.nest.map_structure(
            tf.stop_gradient,
            # Continues to execute the "body" method while 
            # 1) the "cond" is true OR 
            # 2) iteration < maximum_iterations
            tf.while_loop(
                cond=cond,
                body=body,
                maximum_iterations=length,
                loop_vars=[
                    context_output['presents'],
                    context[:, -1],
                    context,
                ],
                shape_invariants=[
                    tf.TensorShape(past_shape(hparams=hparams, batch_size=batch_size)),
                    tf.TensorShape([batch_size]),
                    tf.TensorShape([batch_size, None]),
                ],
            )
        )

        return tokens


class AccumulatingOptimizer(object):
    def __init__(self, opt, var_list):
        self.opt = opt
        self.var_list = var_list
        self.accum_vars = {tv : tf.Variable(tf.zeros_like(tv.read_value()), trainable=False)
                           for tv in var_list}
        self.total_loss = tf.Variable(tf.zeros(shape=[], dtype=tf.float32))
        self.count_loss = tf.Variable(tf.zeros(shape=[], dtype=tf.float32))

    def reset(self):
        updates = [tv.assign(tf.zeros_like(tv)) for tv in self.accum_vars.values()]
        updates.append(self.total_loss.assign(tf.zeros(shape=[], dtype=tf.float32)))
        updates.append(self.count_loss.assign(tf.zeros(shape=[], dtype=tf.float32)))
        with tf.control_dependencies(updates):
            return tf.no_op()

    def compute_gradients(self, loss):
        grads = self.opt.compute_gradients(loss, self.var_list)
        updates = [self.accum_vars[v].assign_add(g) for (g,v) in grads]
        updates.append(self.total_loss.assign_add(loss))
        updates.append(self.count_loss.assign_add(1.0))
        with tf.control_dependencies(updates):
            return tf.no_op()

    def apply_gradients(self):
        grads = [(g,v) for (v,g) in self.accum_vars.items()]
        with tf.control_dependencies([self.opt.apply_gradients(grads)]):
            return self.total_loss / self.count_loss

def load_dataset(enc, path, combine):
    paths = []
    if os.path.isfile(path):
        # Simple file
        paths.append(path)
    elif os.path.isdir(path):
        # Directory
        for (dirpath, _, fnames) in os.walk(path):
            for fname in fnames:
                paths.append(os.path.join(dirpath, fname))
    else:
        # Assume glob
        paths = glob.glob(path)

    token_chunks = []
    raw_text = ''
    for path in tqdm.tqdm(paths):
        if path.endswith('.npz'):
            # Pre-encoded
            with np.load(path) as npz:
                for item in npz.files:
                    token_chunks.append(npz[item])
        elif path.endswith('.csv'):
            start_token = "<|startoftext|>"
            end_token = "<|endoftext|>"
            with open(path, 'r', encoding='utf8', errors='ignore') as fp:
                fp.readline()   # skip header
                reader = csv.reader(fp)
                for row in reader:
                    raw_text += start_token + row[0] + end_token + "\n"
        else:
            # Plain text
            with open(path, 'r', encoding='utf8', errors='ignore') as fp:
                raw_text += fp.read()
            if len(raw_text) >= combine:
                tokens = np.stack(enc.encode(raw_text))
                token_chunks.append(tokens)
                raw_text = ''
            else:
                raw_text += '<|endoftext|>'
    if raw_text:
        tokens = np.stack(enc.encode(raw_text))
        token_chunks.append(tokens)
    return token_chunks

def binary_search(f, lo, hi):
    if f(lo) or not f(hi):
        return None
    while hi > lo + 1:
        mid = (lo + hi) // 2
        if f(mid):
            hi = mid
        else:
            lo = mid
    return hi

class Sampler(object):
    """Fairly samples a slice from a set of variable sized chunks.

    'Fairly' means that the distribution is the same as sampling from one concatenated chunk,
    but without crossing chunk boundaries."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.total_size = sum(chunk.shape[0] for chunk in chunks)
        self.boundaries = [0]
        for i in range(len(chunks)):
            self.boundaries.append(self.boundaries[-1] + chunks[i].shape[0])

    def sample(self, length):
        assert length < self.total_size // len(
            self.chunks
        ), "Dataset files are too small to sample {} tokens at a time".format(
            length)
        while True:
            index = random.randint(0, self.total_size - length - 1)
            i = binary_search(lambda j: self.boundaries[j] > index, 0,
                              len(self.boundaries) - 1) - 1
            if self.boundaries[i + 1] > index + length:
                within_chunk = index - self.boundaries[i]
                return self.chunks[i][within_chunk:within_chunk + length]


def finetune(sess,
             dataset,
             steps=-1,
             model_name='124M',
             model_dir='models',
             combine=50000,
             batch_size=1,
             learning_rate=0.0001,
             accumulate_gradients=5,
             restore_from='latest',
             run_name='run1',
             checkpoint_dir='checkpoint',
             sample_every=100,
             sample_length=512,
             sample_num=1,
             multi_gpu=False,
             save_every=1000,
             print_every=1,
             max_checkpoints=1,
             only_train_transformer_layers=False,
             optimizer='adam',
             overwrite=False,
             reuse=False,
             n_vocab=50257,
             n_ctx=1024,
             n_embd=768,
             n_head=12,
             n_layer=12):
    """Finetunes the model on the given dataset.

    Adapted from https://github.com/nshepperd/gpt-2/blob/finetuning/train.py.
    See that file for parameter definitions.
    """

    # assert model_name not in ['774M', '1558M'] or multi_gpu, "Currently, a modern single GPU cannot finetune the 774M GPT-2 model or larger."


    checkpoint_path = os.path.join(checkpoint_dir, run_name)
    def maketree(path):
        try:
            os.makedirs(path)
        except:
            pass

    maketree(checkpoint_path)
    files = [f for f in os.listdir(checkpoint_path)]
    for file in ['encoder.json', 'vocab.bpe']:
        try:
            shutil.copyfile(os.path.join(model_dir, model_name, file),
                            os.path.join(checkpoint_path, file))
        except FileNotFoundError as fnf_error:
            print("You need to download the GPT-2 model first via download_gpt2()")
            raise(fnf_error)


    enc = get_encoder(checkpoint_path)
    hparams: HParams = HParams(
        n_vocab=n_vocab,
        n_ctx=n_ctx,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
    )
    print(hparams.to_string())

    if sample_length > hparams.n_ctx:
        raise ValueError(
            "Can't get samples longer than window size: %s" % hparams.n_ctx)

    if model_name not in ['117M', '124M']:
        print('For larger models, the recommended finetune() parameters are:')
        print('\tonly_train_transformer_layers = True')
        print('\taccumulate_gradients = 1\n')

    context = tf.compat.v1.placeholder(tf.int32, [batch_size, None])
    gpus = []

    if multi_gpu:
        local_device_protos = device_lib.list_local_devices()
        gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']

    output = build_transformer_model(hparams=hparams, X=context, gpus=gpus, reuse=reuse)
    loss = tf.reduce_mean(
        input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=context[:, 1:], logits=output['logits'][:, :-1]))

    tf_sample = sample_sequence(
        hparams=hparams,
        length=sample_length,
        context=context,
        batch_size=batch_size,
        temperature=1.0,
        top_k=40)

    all_vars = [v for v in tf.compat.v1.trainable_variables() if 'model' in v.name]
    train_vars = [v for v in all_vars if '/h' in v.name] if only_train_transformer_layers else all_vars

    if optimizer == 'adam':
        opt = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)

    if accumulate_gradients > 1:
        opt = AccumulatingOptimizer(
            opt=opt,
            var_list=train_vars)
        opt_reset = opt.reset()
        opt_compute = opt.compute_gradients(loss)
        opt_apply = opt.apply_gradients()
        summary_loss = tf.compat.v1.summary.scalar('loss', opt_apply)
    else:
        opt_grads = tf.gradients(ys=loss, xs=train_vars)
        opt_grads = list(zip(opt_grads, train_vars))
        opt_apply = opt.apply_gradients(opt_grads)
        summary_loss = tf.compat.v1.summary.scalar('loss', loss)

    summary_log = tf.compat.v1.summary.FileWriter(checkpoint_path)

    saver = tf.compat.v1.train.Saver(
        var_list=all_vars,
        max_to_keep=max_checkpoints)
    sess.run(tf.compat.v1.global_variables_initializer())

    if restore_from == 'latest':
        ckpt = tf.train.latest_checkpoint(checkpoint_path)
        if ckpt is None:
            # Get fresh GPT weights if new run.
            ckpt = tf.train.latest_checkpoint(
                os.path.join(model_dir, model_name))
    elif restore_from == 'fresh':
        ckpt = tf.train.latest_checkpoint(
            os.path.join(model_dir, model_name))
    else:
        ckpt = tf.train.latest_checkpoint(restore_from)
    print('Loading checkpoint', ckpt)
    saver.restore(sess, ckpt)

    print('Loading dataset...')
    chunks = load_dataset(enc, dataset, combine)
    data_sampler = Sampler(chunks)
    print('dataset has', data_sampler.total_size, 'tokens')
    print('Training...')

    counter = 1
    counter_path = os.path.join(checkpoint_path, 'counter')
    if os.path.exists(counter_path) and restore_from == 'latest':
        # Load the step number if we're resuming a run
        # Add 1 so we don't immediately try to save again
        with open(counter_path, 'r') as fp:
            counter = int(fp.read()) + 1
    counter_base = counter

    def save():
        maketree(checkpoint_path)
        print(
            'Saving',
            os.path.join(checkpoint_path,
                         'model-{}').format(counter-1))
        saver.save(
            sess,
            os.path.join(checkpoint_path, 'model'),
            global_step=counter-1)
        with open(counter_path, 'w') as fp:
            fp.write(str(counter-1) + '\n')

    def generate_samples():
        SAMPLE_DIR = 'samples'
        context_tokens = data_sampler.sample(1)
        all_text = []
        index = 0
        while index < sample_num:
            out = sess.run(
                tf_sample,
                feed_dict={context: batch_size * [context_tokens]})
            for i in range(min(sample_num - index, batch_size)):
                text = enc.decode(out[i])
                text = '======== SAMPLE {} ========\n{}\n'.format(
                    index + 1, text)
                all_text.append(text)
                index += 1
        print(text)
        maketree(os.path.join(SAMPLE_DIR, run_name))
        with codecs.open(
                os.path.join(SAMPLE_DIR, run_name,
                             'samples-{}').format(counter), 'w', 'utf8') as fp:
            fp.write('\n'.join(all_text))

    def sample_batch():
        return [data_sampler.sample(512) for _ in range(batch_size)]

    if overwrite and restore_from == 'latest':
        for file in files:
            if file.startswith('model') or file.startswith('events'):
                os.remove(os.path.join(checkpoint_path, file))
        save()

    avg_loss = (0.0, 0.0)
    start_time = time.time()

    if steps:
        steps = int(steps)

    try:
        while True:
            if steps > 0 and counter == (counter_base + steps):
                save()
                return
            if (counter - 1) % save_every == 0 and counter > 1:
                save()
            if (counter - 1) % sample_every == 0 and counter > 1:
                generate_samples()

            if accumulate_gradients > 1:
                sess.run(opt_reset)
                for _ in range(accumulate_gradients):
                    sess.run(
                        opt_compute, feed_dict={context: sample_batch()})
                (v_loss, v_summary) = sess.run((opt_apply, summary_loss))
            else:
                (_, v_loss, v_summary) = sess.run(
                    (opt_apply, loss, summary_loss),
                    feed_dict={context: sample_batch()})

            summary_log.add_summary(v_summary, counter)

            if counter % print_every == 0:
                avg_loss = (avg_loss[0] * 0.99 + v_loss,
                            avg_loss[1] * 0.99 + 1.0)

                print(
                    '[{counter} | {time:2.2f}] loss={loss:2.2f} avg={avg:2.2f}'
                    .format(
                        counter=counter,
                        time=time.time() - start_time,
                        loss=v_loss,
                        avg=avg_loss[0] / avg_loss[1]))

            counter += 1
    except KeyboardInterrupt:
        print('interrupted')
        save()





def generate(sess,
             run_name='run1',
             checkpoint_dir='checkpoint',
             model_name=None,
             model_dir='models',
             return_as_list=False,
             truncate=None,
             destination_path=None,
             sample_delim='=' * 20 + '\n',
             prefix=None,
             seed=None,
             nsamples=1,
             batch_size=1,
             length=1023,
             temperature=0.7,
             top_k=0,
             top_p=0.0,
             include_prefix=True):
    """Generates text from a model loaded into memory.

    Adapted from https://github.com/openai/gpt-2/blob/master/src/interactive_conditional_samples.py
    """

    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    if nsamples == 1:
        sample_delim = ''

    if prefix == '':
        prefix = None

    if model_name:
        checkpoint_path = os.path.join(model_dir, model_name)
    else:
        checkpoint_path = os.path.join(checkpoint_dir, run_name)

    enc = get_encoder(checkpoint_path)
    hparams: HParams = HParams()

    if prefix:
        context = tf.compat.v1.placeholder(tf.int32, [batch_size, None])
        context_tokens = enc.encode(prefix)

    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

    output = sample_sequence(
        hparams=hparams,
        length=min(length, length - (len(context_tokens) if prefix else 0)),
        start_token=enc.encoder['<|endoftext|>'] if not prefix else None,
        context=context if prefix else None,
        batch_size=batch_size,
        temperature=temperature, top_k=top_k, top_p=top_p
    )[:, 1:]

    if destination_path:
        f = codecs.open(destination_path, 'w', 'utf-8')
    generated = 0
    gen_texts = []
    while generated < nsamples:
        if not prefix:
            out = sess.run(output)
        else:
            out = sess.run(output, feed_dict={
                    context: batch_size * [context_tokens]
                })
        for i in range(batch_size):
            generated += 1
            gen_text = enc.decode(out[i])
            if prefix:
                gen_text = enc.decode(context_tokens[:1]) + gen_text
            if truncate:
                truncate_esc = re.escape(truncate)
                if prefix and not include_prefix:
                    prefix_esc = re.escape(prefix)
                    pattern = '(?:{})(.*?)(?:{})'.format(prefix_esc,
                                                         truncate_esc)
                else:
                    pattern = '(.*?)(?:{})'.format(truncate_esc)

                trunc_text = re.search(pattern, gen_text, re.S)
                if trunc_text:
                    gen_text = trunc_text.group(1)
            gen_text = gen_text.lstrip('\n')
            if destination_path:
                f.write("{}\n{}".format(gen_text, sample_delim))
            if not return_as_list and not destination_path:
                print("{}\n{}".format(gen_text, sample_delim), end='')
            gen_texts.append(gen_text)

    if destination_path:
        f.close()

    if return_as_list:
        return gen_texts
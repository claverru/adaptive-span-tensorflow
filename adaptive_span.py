import tensorflow as tf
import math


class AdaptiveMask(tf.keras.layers.Layer):
	"""Soft masking function for adaptive size.
	It masks out the last K values of an input. The masking value
	goes from 1 to 0 gradually, so K can be learned with
	back-propagation.
	Args:
		max_size: maximum size (i.e. input dimension)
		ramp_size: size of the ramp going from 0 to 1
		init_val: initial size proportion not to be masked out
		shape: learn multiple sizes independent of each other
	"""
	
	def __init__(self, max_size, ramp_size, init_val=0, shape=(1,)):
		super(AdaptiveMask, self).__init__()
		self._max_size = max_size
		self._ramp_size = ramp_size
		self.init_val = init_val
		self.shape = shape
		self.mask_template = tf.linspace(1. - max_size, 0., num=max_size)
		# self.register_buffer('mask_template', mask_template)

	def build(self, input_shape):
		self.current_val = self.add_weight(
				name='current_val',
				shape=self.shape,
				initializer=tf.keras.initializers.Constant(self.init_val))#,
				# constraint=tf.keras.constraints.MinMaxNorm())
		super(AdaptiveMask, self).build(input_shape)

	def get_current_max_size(self, include_ramp=True):
		current_size = tf.math.ceil(
			tf.reduce_max(self.current_val).numpy() * self._max_size)
		if include_ramp:
			current_size += self._ramp_size
		current_size = tf.maximum(0, min(self._max_size, current_size)).numpy()
		return current_size

	def get_current_avg_size(self, include_ramp=True):
		current_size = tf.math.ceil(
			tf.reduce_mean(self.current_val).numpy() * self._max_size)
		if include_ramp:
			current_size += self._ramp_size
		current_size = tf.maximum(0, min(self._max_size, current_size)).numpy()
		return current_size

	# MinMaxNorm Keras constraint maybe? Can it be applied to non-weights?
	# see above
	def clamp_param(self):
		"""this need to be called after each update"""
		self.current_val.data = tf.clip_by_value(
				self.current_val, 0, 1)

	def call(self, x):
		mask = self.mask_template + self.current_val * self._max_size
		mask = mask / self._ramp_size + 1
		mask = tf.clip_by_value(mask, 0, 1)
		if x.shape[-1] < self._max_size:
			# the input could have been trimmed beforehand to save computation
			mask = mask[:, :, -x.shape[-1]:]
		x = x * mask
		return x


class AdaptiveSpan(tf.keras.layers.Layer):
	"""Adaptive attention span for Transformerself.
	This module learns an attention span length from data for each
	self-attention head.
	Args:
		attn_span: maximum attention span
		adapt_span_loss: loss coefficient for the span length
		adapt_span_ramp: length of the masking ramp
		adapt_span_init: initial size ratio
		adapt_span_cache: adapt cache size to reduce memory usage
	"""
	def __init__(self, attn_span, adapt_span_loss, adapt_span_ramp,
				 adapt_span_init, adapt_span_cache, nb_heads, **kargs):
		super(AdaptiveSpan, self).__init__()
		self._adapt_cache = adapt_span_cache
		self._max_span = attn_span
		self._loss_coeff = adapt_span_loss
		self._nb_heads = nb_heads

		self._adapt_span_ramp = adapt_span_ramp
		self._adapt_span_init = adapt_span_init

	def build(self, input_shape):
		self._mask = AdaptiveMask(
				max_size=self._max_span,
				ramp_size=self._adapt_span_ramp,
				init_val=self._adapt_span_init,
				shape=(self._nb_heads, 1, 1))
		super(AdaptiveSpan, self).build(input_shape)

	def get_trim_len(self):
		"""how much of memory can be trimmed to reduce computation"""
		L = self._max_span
		trim_len = min(L - 1, L - self._mask.get_current_max_size())
		# too fine granularity might be bad for the memory management
		trim_len = tf.floor(trim_len / 64) * 64
		return trim_len.numpy().item()

	def trim_memory(self, query, key, value, key_pe):
		"""trim out unnecessary memory beforehand to reduce computation"""
		trim_len = self.get_trim_len()
		cache_size = key.shape[1] - query.shape[1]
		trim_len_cache = trim_len - (self._max_span - cache_size)
		if trim_len_cache > 0:
			key = key[:, trim_len_cache:, :]
			value = value[:, trim_len_cache:, :]
		elif trim_len_cache < 0:
			# cache is too short! this happens when validation resumes
			# after a lot of updates.
			key = tf.pad(key, [[-trim_len_cache, 0], [0, 0]])
			value = tf.pad(value, [[-trim_len_cache, 0], [0, 0]])
		if trim_len > 0:
			if key_pe is not None:
				key_pe = key_pe[:, :, trim_len:]
		return key, value, key_pe

	def get_cache_size(self):
		"""determine how long the cache should be"""
		if self._adapt_cache:
			trim_len = self.get_trim_len()
			# give a buffer of 64 steps since a span might increase
			# in future updates
			return min(self._max_span, self._max_span - trim_len + 64)
		else:
			return self._max_span

	# keras regularizers?
	def get_loss(self):
		"""a loss term for regularizing the span length"""
		return self._loss_coeff * self._max_span \
				* tf.reduce_mean(self._mask.current_val)

	def get_current_max_span(self):
		return self._mask.get_current_max_size()

	def get_current_avg_span(self):
		return self._mask.get_current_avg_size()

	def clamp_param(self):
		self._mask.clamp_param()

	def call(self, attn):
		B = attn.shape[0]
		M = attn.shape[1]
		attn = tf.reshape(attn, (B // self._nb_heads, self._nb_heads, M, -1))

		attn = self._mask(attn)
		attn /= tf.reduce_sum(attn, axis=-1, keepdims=True) + 1e-8

		attn = tf.reshape(attn, (B, M, -1))

		return attn

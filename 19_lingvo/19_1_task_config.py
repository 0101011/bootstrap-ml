def Task(cls):
  p = model.AsrModel.Params()
  p.name = 'librispeech'

  # Initialize encoder params.
  ep = p.encoder
  
  # Data consists 240 dimensional frames (80 x 3 frames), which we
  # re-interpret as individual 80 dimensional frames. See also,
  # LibrispeechCommonAsrInputParams.
  ep.input_shape = [None, None, 80, 1]
  ep.lstm_cell_size = 1024
  ep.num_lstm_layers = 4
  ep.conv_filter_shapes = [(3, 3, 1, 32), (3, 3, 32, 32)]
  ep.conv_filter_strides = [(2, 2), (2, 2)]
  ep.cnn_tpl.params_init = py_utils.WeightInit.Gaussian(0.001)
  
  # Disable conv LSTM layers.
  ep.num_conv_lstm_layers = 0

  # Initialize decoder params.
  dp = p.decoder
  dp.rnn_cell_dim = 1024
  dp.rnn_layers = 2
  dp.source_dim = 2048
  # Use functional while based unrolling.
  dp.use_while_loop_based_unrolling = False

  tp = p.train
  tp.learning_rate = 2.5e-4
  tp.lr_schedule = lr_schedule.ContinuousLearningRateSchedule.Params().Set(
      start_step=50000, half_life_steps=100000, min=0.01)

  # Setting p.eval.samples_per_summary to a large value ensures that dev,
  # devother, test, testother are evaluated completely (since num_samples for
  # each of these sets is less than 5000), while train summaries will be
  # computed on 5000 examples.
  p.eval.samples_per_summary = 5000
  p.eval.decoder_samples_per_summary = 0

  # Use variational weight noise to prevent overfitting.
  p.vn.global_vn = True
  p.train.vn_std = 0.075
  p.train.vn_start_step = 20000

return p

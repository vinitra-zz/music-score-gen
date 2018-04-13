# music-score-gen

Current Progress 4/13

Vinitra started to look at downloading the data and experimenting with the Youtube 8M dataset.
Here's the tutorial she's following: https://github.com/google/youtube-8m

'curl data.yt8m.org/download.py | shard=1,100 partition=1/video_level/train mirror=us python' should download 1/100th of the 31GB video level dataset, but Vinitra's laptop is being weird and there's an error message asking for 2TB of free disk space.

If you have compute, try that after cding into features?

## Re: Music Generation
https://arxiv.org/pdf/1709.01620.pdf
Skimmed through some parts of this *book*(?!) to familiarize myself with the problem space more
https://github.com/tensorflow/magenta

### Using Docker seems to be the easiest way to go
Takes a long time to start...
docker run -it -p 6006:6006 -v /tmp/magenta:/magenta-data tensorflow/magenta
"""Remember that only /magenta-data will persist"""
melody_rnn_generate \
  --config=lookback_rnn \
  --bundle_file=/magenta-models/lookback_rnn.mag \
  --output_dir=/magenta-data/lookback_rnn/generated \
  --num_outputs=10 \
  --num_steps=128 \
  --primer_melody="[60]"

### Playing MIDI files is confusing
After generating in Docker container, you can only play on host machine
Here's what tensorflow magenta has to say:
https://github.com/tensorflow/magenta/tree/master/magenta/interfaces/midi
They also say AI Jam is easier?
https://github.com/tensorflow/magenta-demos/tree/master/ai-jam-js

## Baseline Model
I think if we use the music generation from Tensorflow Magenta with the "genre" of the videos as classified by Youtube 8M, we have baseline generated content?

List of models + papers could be useful for baseline: https://github.com/tensorflow/magenta/tree/master/magenta/models

Not too sure what this paper is about (haven't read yet) but is a Magenta model and generates artistic representations from images? https://arxiv.org/abs/1610.07629
Image Stylization: A "Multistyle Pastiche Generator" that generates artistics representations of photographs. Described in A Learned Representation For Artistic Style.


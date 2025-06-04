clc;
clear;
close all;

%% Step 1: Load and Preprocess Voice Signal
[voice, fs] = audioread('audio.opus');  % Replace with your speech file

voice = voice(:,1);              % Mono channel
voice = resample(voice, 8000, fs); % GSM uses 8 kHz
fs = 8000;
sound(voice, fs);
disp('Original speech playing...');
samples_per_symbol = 8;  % Number of samples per GMSK symbol
BT = 0.3;                % Bandwidth-Time product for Gaussian filter
snr = 10; % Signal-to-Noise Ratio in dB


%% Step 2: Normalize and Quantize
voice_norm = (voice - min(voice)) / (max(voice) - min(voice)); % Normalize to [0,1]
voice_bin = de2bi(uint8(voice_norm * 255), 8, 'left-msb');      % 8-bit binary
voice_bitstream = reshape(voice_bin.', 1, []);                  % Serial bitstream

%% Step 3: Convolutional Encoding (Rate 1/2)
trellis = poly2trellis(7, [171 133]); % GSM-standard convolutional encoder
coded_bits = convenc(voice_bitstream, trellis);

%% Step 4: Interleaving (Block-based with permutation)
burst_size = 456;
num_bits = length(coded_bits);
num_full_bursts = floor(num_bits / burst_size);
coded_bits = coded_bits(1 : num_full_bursts * burst_size); % Trim to fit bursts

% Reshape and interleave
interleaved_matrix = reshape(coded_bits, burst_size, []);
perm = randperm(burst_size);                 % Store this for deinterleaving
interleaved_matrix = interleaved_matrix(perm, :);
interleaved_bits = interleaved_matrix(:).';

%% Step 5: Custom GMSK Modulation (phase shaping)
Tb = 1;  % Bit duration
BT = 0.3;
Fs = samples_per_symbol / Tb;  % Sampling frequency
h = 0.5;  % Modulation index for GMSK

% NRZ encoding: 0 -> -1, 1 -> +1
nrz = 2 * interleaved_bits - 1;

% Gaussian filter
BT_product = BT;
span = 4;  % Filter span in symbols
sps = samples_per_symbol;
B = BT_product / Tb;
alpha = sqrt(log(2)/2) / (B*Tb); % Gaussian filter std dev

t = linspace(-span/2, span/2, span*sps);
g_filter = exp(-2 * pi^2 * alpha^2 * t.^2);
g_filter = g_filter / sum(g_filter); % Normalize

% Phase shaping
phase = pi * h * cumsum(upsample(nrz, sps));
filtered_phase = conv(phase, g_filter, 'same');
modSignal = exp(1j * filtered_phase); % Complex baseband

%% Step 6: Channel (AWGN)
rxSignal = awgn(modSignal, snr, 'measured');

%% Step 7: Custom GMSK Demodulation (Differential Phase Detection)
rx_phase = unwrap(angle(rxSignal(:)));  % Force column vector

diff_phase = [0; diff(rx_phase)];
% Threshold to get binary stream
demodBits = double(diff_phase > 0);
demodBits = demodBits(1:sps:end); % Downsample
demodBits = demodBits(1:length(interleaved_bits)); % Match length


%% Step 8: Deinterleaving
demodBits = demodBits(1 : num_full_bursts * burst_size); % Trim excess
demod_matrix = reshape(demodBits, burst_size, []);
[~, inv_perm] = sort(perm);
demod_matrix = demod_matrix(inv_perm, :);
demodBits = demod_matrix(:).';

%% Step 9: Viterbi Decoding
tblen = 34;
decoded_bits = vitdec(demodBits, trellis, tblen, 'trunc', 'hard');

%% Step 10: Reconstruct Voice
% Trim to nearest byte
decoded_bits = decoded_bits(1:floor(length(decoded_bits)/8)*8);
voice_uint8 = bi2de(reshape(decoded_bits, 8, []).', 'left-msb');
voice_rec = double(voice_uint8) / 255;
voice_rec = 2 * (voice_rec - 0.5); % Back to [-1, 1]
sound(voice_rec, fs);
disp('Recovered speech playing...');

%% Step 11: BER Calculation and Plot
[ber, numErrors] = biterr(voice_bitstream(1:length(decoded_bits)), decoded_bits);
fprintf('Bit Error Rate (BER): %.4f\n', ber);

% Plot comparison
figure;
subplot(2,1,1);
plot(voice);
title('Original Speech Signal');
xlabel('Sample'); ylabel('Amplitude');

subplot(2,1,2);
plot(voice_rec);
title('Recovered Speech Signal');
xlabel('Sample'); ylabel('Amplitude');

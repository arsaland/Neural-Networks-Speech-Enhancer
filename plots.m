% ---------------------------------------------------------
% clean command windows

clear all; close all; clc;
warning off

% ---------------------------------------------------------
% define inputs

OFILE = 'o.wav';
NFILE = 'n.wav';

SNR  = 5; % in dBs
CLIP = 500;


% ---------------------------------------------------------
% read audio signals

% original signal
[osig, ofs] = audioread(OFILE);

% noisy signal
[nsig, nfs] = audioread(NFILE);


% ---------------------------------------------------------
% set/update noisy signal with desired SNR

nsig1 = set_noise(osig, nsig, 5);
nsig2 = set_noise(osig, nsig, 0);
disp(sprintf('Original SNR=%.2f', snr(osig, nsig-osig)))


% ---------------------------------------------------------

figure()

subplot(3,1,1)
plot(osig)
title('Original signal')

subplot(3,1,2)
plot(nsig1)
title('Noisy signal with SNR=5 dB')

subplot(3,1,3)
plot(nsig2)
title('Noisy signal with SNR=0 dB')


% ---------------------------------------------------------

[osp, of] = get_spectrum(osig, ofs);
[nsp, nf] = get_spectrum(nsig, nfs);

% Plot the spectrum:

figure();

subplot(2,1,1)
plot(of, osp);
xlabel('Frequency (Hz)');
title('Frequency Spectrum for original signal');

subplot(2,1,2)
plot(nf, nsp);
xlabel('Frequency (Hz)');
title('Frequency Spectrum for noisy signal');



% ---------------------------------------------------------
% functions

function [SIG, F] = get_spectrum(sig, fs)
	N = length(sig); dt = 1/fs;
	t = (0:dt:(N-1)*dt)';

	% Fourier Transform:
	SIG = fftshift(fft(sig));
	SIG = abs(SIG)/N;

	%Frequency specifications:
	dF = fs/N; % hertz
	F = -fs/2:dF:fs/2-dF;
end

function labels = get_labels(clipped, th)
	labels = std(clipped) < th;
	labels = onehotencode(categorical(labels), 1);
end

function features = get_features(clipped)
	nsamples = size(clipped, 2);
	features = zeros(5, nsamples);

	features(1, :) = get_zerocrossing(clipped);
	features(2, :) = rms(clipped);
	features(3, :) = std(clipped);
	features(4, :) = max(clipped);
	features(5, :) = get_signal_avg_power(clipped);
end

function zc = get_zerocrossing(clipped)
	nsamples = size(clipped, 2);
	CLIP     = size(clipped, 1);
	zc = zeros(1, nsamples);

	for s = 1:nsamples
		sig = clipped(:, s);
		c   = find( sig(1:end-1).*sig(2:end) < 0 );
		zc(s)  = length(c); % / CLIP;
	end
end

function clipped = clip_signal(sig, CLIP)
	sig = reshape(sig, [], 1);
	sig_len = length(sig);
	windows = fix(sig_len/CLIP);

	clipped = reshape(sig(1:CLIP*windows), CLIP, windows);
end

function nsig = set_noise(osig, nsig, SNR)
	% original noise
	noise = nsig - osig;

	% get energy for original signal
	Eosig = get_signal_energy(osig);

	% get energy for original signal
	Enoise = get_signal_energy(noise);

	% update noise with desired SNR
	noise = noise * sqrt( Eosig / ( 10^(SNR/10) * Enoise ) );

	% update noisy signal with updated noise
	nsig = osig + noise;
end

function E = get_signal_energy(sig)
	E = sig' * sig;
end

function P = get_signal_avg_power(sig)
	P = rms(sig).^2;
end
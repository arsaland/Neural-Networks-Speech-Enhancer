% ---------------------------------------------------------
% clean command windows

clear all; close all; clc;
warning off

% ---------------------------------------------------------
% define inputs

OFILE = 'o.wav';
NFILE = 'n.wav';

SNR  = 0; % in dBs
CLIP = 500;


% ---------------------------------------------------------
% load neural network

NN_FILE = sprintf('net_CLIP=%d_SNR=%.2f.mat', CLIP, SNR);
net = load(NN_FILE);
net = net.net;


% ---------------------------------------------------------
% read audio signals

% original signal
[osig, ofs] = audioread(OFILE);

% noisy signal
[nsig, nfs] = audioread(NFILE);

% ---------------------------------------------------------
% set/update noisy signal with desired SNR

nsig = set_noise(osig, nsig, SNR);
input_snr = snr(osig, nsig-osig);
disp(sprintf('Original SNR=%.2f', input_snr))


% ---------------------------------------------------------
% prepare input data

nclipped = clip_signal(nsig, CLIP);
oclipped = clip_signal(osig, CLIP);
features = get_features(nclipped);


% ---------------------------------------------------------
% evaluation

p = round( net( features ) );
p = onehotdecode(p,categories(categorical(p)),1);
p = logical(double(p)-1);

small_value = 0.4; 
fade_samples = 500;  % or some other number of samples suitable for your data

fade_in = linspace(small_value, 1, fade_samples);
fade_out = linspace(1, small_value, fade_samples);

for s = 1:size(nclipped,2)
    noise = nclipped(:,s) - oclipped(:,s);
	if p(s)
        nclipped(1:fade_samples, s) = nclipped(1:fade_samples, s) .* fade_out';
        nclipped(end-fade_samples+1:end, s) = nclipped(end-fade_samples+1:end, s) .* fade_in';
        nclipped(:,s) = nclipped(:,s) * small_value;
	else
        sig = nclipped(:,s);
		nclipped(:,s) = spectral_subtraction(sig, noise, nfs);
	end
end

out_sig = nclipped(:);
out_snr = snr( osig( 1:length( out_sig) ), out_sig-osig( 1:length( out_sig) ) );
disp(sprintf('Output SNR=%.2f', out_snr))

figure()

subplot(3,1,1)
plot(osig)
title(sprintf('Clean signal', osig))

subplot(3,1,2)
plot(nsig)
title(sprintf('Noisy signal with SNR=%.2fdB', input_snr))

subplot(3,1,3)
plot(out_sig)
title(sprintf('Enhanced signal with SNR=%.2fdB using Proposed Method', out_snr))

sound(out_sig)
audiowrite('output.wav', out_sig, ofs);
% 
% 
% 
% 
% % Prepare Spectrogram Windows
% figure('Name', 'Spectrograms of Signals');
% 
% % Plot the Spectrogram of the original clean signal
% subplot(3,1,1);
% spectrogram(osig, 256, 250, 256, ofs, 'yaxis');
% title(sprintf('Clean signal', input_snr));
% xlabel('Time (s)');
% ylabel('Frequency (Hz)');
% colormap('parula');
% 
% % Plot the Spectrogram of the noisy signal
% subplot(3,1,2);
% spectrogram(nsig, 256, 250, 256, nfs, 'yaxis');
% title(sprintf('Noisy signal with SNR=%.2fdB', input_snr));
% xlabel('Time (s)');
% ylabel('Frequency (Hz)');
% colormap('parula');
% 
% % Plot the Spectrogram of the enhanced/output signal
% subplot(3,1,3);
% spectrogram(out_sig, 256, 250, 256, ofs, 'yaxis');
% title(sprintf('Enhanced signal with SNR=%.2fdB using Proposed Method', out_snr));
% xlabel('Time (s)');
% ylabel('Frequency (Hz)');
% colormap('parula');

% ---------------------------------------------------------
%Additional PSD Plotting
% 
% figure('Name', 'Power Spectral Density of Signals');
% 
% % Formant Tracking Plotting
% figure('Name', 'Formant Tracks of Signals');
% 
% subplot(3,1,1);
% plot_formant_tracks(osig, ofs);
% title('Formant Tracks of Clean Signal');
% 
% subplot(3,1,2);
% plot_formant_tracks(nsig, nfs);
% title(sprintf('Formant Tracks of Noisy Signal - SNR = %.2fdB', input_snr));
% 
% subplot(3,1,3);
% plot_formant_tracks(out_sig, ofs);
% title(sprintf('Formant Tracks of Enhanced Signal - SNR = %.2fdB', out_snr));

% % ------------------------ formants
% plot_formant_tracks(osig, nsig, out_sig, ofs);
% 
% 
% function formants = get_formants(frame, fs)
%     % Calculate LPC coefficients
%     a = lpc(frame, 8); % You can change the order of LPC
%     
%     % Find the roots of the LPC polynomial
%     r = roots(a);
%     
%     % Find angle of roots
%     angz = atan2(imag(r), real(r));
%     
%     % Convert to Hz
%     formants = angz * (fs / (2 * pi));
%     
%     % Keep only positive frequencies
%     formants = formants(formants > 0);
%     
%     % Sort formants
%     formants = sort(formants);
%     
%     % Ensure that only the first three formants are returned
%     if length(formants) > 3
%         formants = formants(1:3);
%     end
% end
% 
% function plot_formant_tracks(clean_signal, noisy_signal, enhanced_signal, fs)
%    frame_size = 256;
%     overlap = frame_size / 2;
%     num_formants = 3; % change this as per your requirement
% 
%     % Number of frames
%     num_frames = floor(length(clean_signal) / overlap) - 1;
% 
%     % Initialize matrices to store formant frequencies for each frame.
%     clean_formants = zeros(num_formants, num_frames);
%     noisy_formants = zeros(num_formants, num_frames);
%     enhanced_formants = zeros(num_formants, num_frames);
% 
%     for idx = 1:num_frames
%         start_idx = (idx - 1) * overlap + 1;
%         end_idx = start_idx + frame_size - 1;
%         
%         if end_idx > length(clean_signal)
%             break; % Exit loop if end_idx exceeds the length of the signal
%         end
%         clean_frame = clean_signal(start_idx:end_idx);
%         
%         if end_idx > length(noisy_signal)
%             break; % Exit loop if end_idx exceeds the length of the signal
%         end
%         noisy_frame = noisy_signal(start_idx:end_idx);
%         
%         if end_idx > length(enhanced_signal)
%             break; % Exit loop if end_idx exceeds the length of the signal
%         end
%         enhanced_frame = enhanced_signal(start_idx:end_idx);
% 
%         clean_formants(:, idx) = get_formants(clean_frame, fs);
%         noisy_formants(:, idx) = get_formants(noisy_frame, fs);
%         enhanced_formants(:, idx) = get_formants(enhanced_frame, fs);
%     end    
%     time = ((1:num_frames) - 1) * overlap / fs;
%     
%      for formant_idx = 1:num_formants
%         figure('Name', sprintf('Formant %d', formant_idx));
%         
%         clean_formant = clean_formants(formant_idx, :);
%         noisy_formant = noisy_formants(formant_idx, :);
%         enhanced_formant = enhanced_formants(formant_idx, :);
%         
%         % Compute the MSE between the clean and noisy formants
%         mse_clean_noisy = mean((clean_formant - noisy_formant).^2, 'omitnan');
%         
%         % Compute the MSE between the clean and enhanced formants
%         mse_clean_enhanced = mean((clean_formant - enhanced_formant).^2, 'omitnan');
%         
%         % Plots
%         subplot(2,1,1);
%         plot(time, clean_formant, 'b');
%         hold on;
%         plot(time, noisy_formant, 'r');
%         xlabel('Time (s)');
%         ylabel('Frequency (Hz)');
%         title(sprintf('Formant %d: Clean (Blue) and Noisy (Red) Signal - MSE=%.4f', formant_idx, mse_clean_noisy));
%         legend('Clean', 'Noisy');
%         hold off;
%         
%         subplot(2,1,2);
%         plot(time, clean_formant, 'b');
%         hold on;
%         plot(time, enhanced_formant, 'g');
%         xlabel('Time (s)');
%         ylabel('Frequency (Hz)');
%         title(sprintf('Formant %d: Clean (Blue) and Enhanced (Green) Signal using Proposed Method- MSE=%.4f', formant_idx, mse_clean_enhanced));
%         legend('Clean', 'Enhanced');
%         hold off;
%         
% 
%     end
% end

% ---------------------------------------------------------
% % Final PSD Plotting
% 
% figure('Name', 'Power Spectral Density of Signals');
% 
% % Create the first subplot for clean and noisy signals
% subplot(2,1,1); % 2 rows, 1 column, 1st subplot
% 
% % Compute and plot the PSD of the original clean signal
% [pxx_clean, f] = pwelch(osig, 256, [], [], ofs); 
% plot(f,10*log10(pxx_clean), 'b'); % 'b' specifies blue color
% hold on; % Retain current plot when adding new plots
% 
% % Compute and plot the PSD of the noisy signal
% [pxx_noisy, f] = pwelch(nsig, 256, [], [], nfs); 
% plot(f,10*log10(pxx_noisy), 'r'); % 'r' specifies red color
% 
% % Compute MSE between clean and noisy signal in this part of the spectrum
% mse_clean_noisy = immse(osig, nsig);
% 
% title(sprintf('Power Spectral Density of Clean (Blue) and Noisy (Red) Signals - MSE=%.4f', mse_clean_noisy));
% xlabel('Frequency (Hz)');
% ylabel('Power/Frequency (dB/Hz)');
% 
% % Add a legend to distinguish the lines
% legend('Clean Signal', sprintf('Noisy Signal', input_snr));
% 
% hold off; % Release the hold on the first subplot
% 
% % Create the second subplot for clean and enhanced signals
% subplot(2,1,2); % 2 rows, 1 column, 2nd subplot
% 
% % Compute and plot the PSD of the original clean signal
% [pxx_clean, f] = pwelch(osig, 256, [], [], ofs); 
% plot(f,10*log10(pxx_clean), 'b'); % 'b' specifies blue color
% hold on; % Retain current plot when adding new plots
% 
% % Compute and plot the PSD of the enhanced signal
% [pxx_enhanced, f] = pwelch(out_sig, 256, [], [], ofs); 
% plot(f,10*log10(pxx_enhanced), 'g'); % 'g' specifies green color
% 
% % Compute MSE between clean and enhanced signal in this part of the spectrum
% % resize osig to match the size of out_sig
% osig_resized = osig(1:length(out_sig));
% mse_clean_enhanced = immse(osig_resized, out_sig);
% 
% title(sprintf('Power Spectral Density of Clean (Blue) and Enhanced (Green) Signals using Proposed Method - MSE=%.4f', mse_clean_enhanced));
% xlabel('Frequency (Hz)');
% ylabel('Power/Frequency (dB/Hz)');
% 
% % Add a legend to distinguish the lines
% legend('Clean Signal', sprintf('Enhanced Signal', out_snr));
% 
% hold off; % Release the hold on the second subplot


% ---------------------------------------------------------
% functions

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

function sig_ss = spectral_subtraction(sig, noise, fs)
    % perform fft
    sig_fft = fft(sig);
    noise_fft = fft(noise);

    % calculate the spectral subtraction
    sig_ss_fft = abs(sig_fft) - abs(noise_fft);

    % ensure non-negative
    sig_ss_fft(sig_ss_fft < 0) = 0;

    % retain the phase of the original signal
    sig_ss_fft = sig_ss_fft .* exp(1i * angle(sig_fft));

    % perform ifft
    sig_ss = real(ifft(sig_ss_fft));
end

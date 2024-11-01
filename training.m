% ---------------------------------------------------------
% clean command windows

clear all; close all; clc;
warning off

% ---------------------------------------------------------
% define inputs

OFOLDER = 'clean';
NFOLDER = 'noisy';

SNR  = 25; % in dBs
CLIP = 500;

% ---------------------------------------------------------
% read audio signals

olist = dir(fullfile(OFOLDER, '*.wav'));
nlist = dir(fullfile(NFOLDER, '*.wav'));
nwavs = length(olist);

osig = []; nsig = [];

for n = 1:nwavs

	% path to current wav files
	OFILE = fullfile(olist(n).folder, olist(n).name);
	NFILE = fullfile(nlist(n).folder, nlist(n).name);

	% read current original signal
	[osig_temp, ofs] = audioread(OFILE);
	
	% read current noisy signal
	[nsig_temp, nfs] = audioread(NFILE);
	
	% set/update noisy signal with desired SNR
	nsig_temp = set_noise(osig_temp, nsig_temp, SNR);

	% store this signal to train later
	osig = [osig; osig_temp];
	nsig = [nsig; nsig_temp];
end

% ---------------------------------------------------------
% prepare training data

oclipped  = clip_signal(osig, CLIP);
nclipped  = clip_signal(nsig, CLIP);
features = get_features(nclipped);
labels   = get_labels(oclipped, 0.01);


% ---------------------------------------------------------
% training step

hiddenLayerSize = 10;
net = patternnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100; 
net.divideParam.valRatio = 15/100; 
net.divideParam.testRatio = 15/100; 
net.trainFcn = 'trainscg';
% net.trainParam.showCommandLine = true;
% net.trainParam.showWindow = false;
[net,tr] = trainrp(net, features, labels);

% save the network
NN_FILE = sprintf('net_CLIP=%d_SNR=%.2f.mat', CLIP, SNR);
save(NN_FILE, 'net');



% ---------------------------------------------------------
% functions

function labels = get_labels(clipped, th)
	labels = std(clipped) < th;
	labels = onehotencode(categorical(labels), 1);
end

function features = get_features(clipped)
	nsamples = size(clipped, 2); % gives number of columns
	features = zeros(5, nsamples);

	features(1, :) = get_zerocrossing(clipped);
	features(2, :) = rms(clipped);
	features(3, :) = std(clipped);
	features(4, :) = max(clipped);
	features(5, :) = get_signal_avg_power(clipped);
end

function zc = get_zerocrossing(clipped)
	nsamples = size(clipped, 2); % gives number of columns
	CLIP     = size(clipped, 1); % gives number of rows
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

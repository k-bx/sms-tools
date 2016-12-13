import numpy as np
import os
import soundDownload as SD
import soundAnalysis as SA
import essentia.standard as ess
from essentia import Pool
from numpy import mean
from scipy.cluster.vq import kmeans, whiten
import matplotlib.pyplot as plt


def main():
    # Part 1
    # download_sounds()

    # Part 2
    # cluster()

    # Part 3.1
    # n_clusters, info_arr, centroids, feature_arr_wht = cluster_essentia()
    # output_results(n_clusters, info_arr, centroids, feature_arr_wht)

    # Part 3.2 -- stripping silences
    # plot_coefficients(M=1024, N=1024, H=512)
    n_clusters, info_arr, centroids, feature_arr_wht = cluster_essentia(strip=False)
    output_results(n_clusters, info_arr, centroids, feature_arr_wht)


def download_sounds():
    sounds = [
        ('violin', 'multisample', (0, 5)),
        ('guitar', 'multisample', (0, 5)),
        ('bassoon', 'multisample', (0, 5)),
        ('trumpet', 'multisample', (0, 5)),
        ('clarinet', 'multisample', (0, 5)),
        ('cello', 'multisample', (0, 5)),
        ('xiaoluo', None, (0, 5)),
        ('daluo', None, (0, 5)),
        ('flute', 'multisample', (0, 5)),
        ('mridangam', None, (0, 5))
        ]
    for (text, tag, dur) in sounds:
        SD.downloadSoundsFreesound(queryText=text,
                                   API_Key='kVuI0F9Qf9UbqXGYVx7TlAnF57nujpT3K9UtbCKb',
                                   outputDir='tmp/', topNResults=20,
                                   duration=dur, tag=tag)


def get_paths():
    """
    Returns a list of tuples of category, sound id and full path
    """
    inputDir = 'tmp'
    descExt = '.mp3'
    res = []
    for path, dname, fnames in os.walk(inputDir):
        for fname in fnames:
            if descExt in fname.lower():
                remain, rname, cname, sname = (path.split('/')[:-3],
                                               path.split('/')[-3],
                                               path.split('/')[-2],
                                               path.split('/')[-1])
                path = os.path.join('/'.join(remain), rname, cname,
                                    sname, fname)
                res.append((cname, sname, path))
    return res


def cluster():
    SA.showDescriptorMapping()
    SA.clusterSounds('tmp/', nCluster=10, descInput=[0, 3])


def cluster_essentia(strip=False):
    n_clusters = 10
    feature_arr = []
    paths = get_paths()
    info_arr = []
    for cname, sname, path in paths:
        feature_arr.append(get_sound_features(path, strip=strip))
        info_arr.append([sname, cname])
    feature_arr = np.array(feature_arr)
    info_arr = np.array(info_arr)
    feature_arr_wht = whiten(feature_arr)
    centroids, distortion = kmeans(feature_arr_wht, n_clusters)
    return n_clusters, info_arr, centroids, feature_arr_wht


def get_sound_features(path, strip=False):
    """
    Returns a list of features for a particular sound as doubles.
    """
    audio = ess.MonoLoader(filename=path)()
    if strip:
        audio = strip_sound(audio, threshold=0.1)
    M = 1024
    N = 1024
    H = 512
    w = ess.Windowing(type='hann', size=M)
    spec = ess.Spectrum()
    pool = Pool()

    features = [
        ('SpectralCentroidTime',
         ess.SpectralCentroidTime()),
        ('LogAttackTime',
         lambda x: ess.LogAttackTime()(x)[0]),
        ('SpectralContrast.mean',
         lambda x: mean(ess.SpectralContrast(frameSize=N)(x)[0])),
        ('MFCC',
         lambda x: mean(ess.MFCC(inputSize=int(N / 2 + 1))(x)[1])),
        ('RollOff',
         lambda x: ess.RollOff()(x)),
    ]

    for frame in ess.FrameGenerator(audio, frameSize=N, hopSize=H):
        spec_res = spec(w(frame))
        for (feature_name, feature) in features:
            feature_res = feature(spec_res)
            pool.add(feature_name, feature_res)

    # stats = ['mean', 'median']
    stats = ['mean', 'var', 'min', 'max']
    aggrpool = ess.PoolAggregator(defaultStats=stats)(pool)

    features = []
    for descriptor_name in aggrpool.descriptorNames():
        features.append(aggrpool[descriptor_name])

    return features


def output_results(n_clusters, info_arr, centroids, feature_arr_wht):
    # Rest is mostly copy-paste from soundAnalysis.py
    clusResults = -1*np.ones(feature_arr_wht.shape[0])
    for ii in range(feature_arr_wht.shape[0]):
        diff = centroids - feature_arr_wht[ii, :]
        diff = np.sum(np.power(diff, 2), axis=1)
        indMin = np.argmin(diff)
        clusResults[ii] = indMin

    classCluster = []
    globalDecisions = []
    for ii in range(n_clusters):
        ind = np.where(clusResults == ii)[0]
        freqCnt = []
        if len(ind) > 0:
            for elem in info_arr[ind, 1]:
                freqCnt.append(info_arr[ind, 1].tolist().count(elem))
        else:
            pass
        indMax = np.argmax(freqCnt)
        classCluster.append(info_arr[ind, 1][indMax])
        decisions = []
        for jj in ind:
            if info_arr[jj, 1] == classCluster[-1]:
                decisions.append(1)
            else:
                decisions.append(0)
        globalDecisions.extend(decisions)
    globalDecisions = np.array(globalDecisions)
    totalSounds = len(globalDecisions)
    nIncorrectClassified = len(np.where(globalDecisions == 0)[0])
    print("Out of %d sounds, %d sounds are incorrectly classified considering that one cluster should "
          "ideally contain sounds from only a single class" % (totalSounds, nIncorrectClassified))
    print("You obtain a classification (based on obtained clusters and majority voting) accuracy "
          "of %.2f percentage" % round(float(100.0*float(totalSounds - nIncorrectClassified) / totalSounds), 2))


def plot_coefficients(M, N, H):
    """Temporary code for plotting"""
    # path = 'tmp/bassoon/154350/154350_2626346-hq.mp3'
    # path = 'tmp/cello/42251/42251_7037-hq.mp3'
    path = 'tmp/clarinet/248700/248700_6552981-hq.mp3'
    audio = ess.MonoLoader(filename=path)()
    energy = ess.Energy()
    sound_energy = []
    for frame in ess.FrameGenerator(audio, frameSize=N, hopSize=H):
        e = energy(frame)
        sound_energy.append(e)
    plt.plot(sound_energy)
    plt.show()


def strip_sound(audio, threshold):
    M = 1024
    N = 1024
    H = 512
    counting_start = True
    start_frame = 0
    end_frame = 0
    energy = ess.Energy()
    for frame in ess.FrameGenerator(audio, frameSize=N, hopSize=H):
        e = energy(frame)
        if counting_start:
            if e < threshold:
                start_frame = start_frame + 1
            else:
                counting_start = False
                end_frame = start_frame
        else:
            if e > threshold:
                end_frame = end_frame + 1
            else:
                break
    new_audio = audio[H * start_frame:H * end_frame]
    return np.array(new_audio)


if __name__ == '__main__':
    main()

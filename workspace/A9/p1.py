import soundDownload as SD
import soundAnalysis as SA


if __name__ == '__main__':
    # sounds = [('clarinet', 'multisample', (0, 10)),
    #           ('cello', 'multisample', (0, 10)),
    #           ('naobo', None, (0, 10))]
    # sounds = [('violin', None, (0, 3))]
    # sounds = [('oboe', None, (0, 3))]
    # for (text, tag, dur) in sounds:
    #     SD.downloadSoundsFreesound(queryText=text,
    #                                API_Key='kVuI0F9Qf9UbqXGYVx7TlAnF57nujpT3K9UtbCKb',
    #                                outputDir='tmp/', topNResults=20,
    #                                duration=dur, tag=tag)

    # SA.showDescriptorMapping()
    # SA.descriptorPairScatterPlot('tmp/', descInput=(10, 5))

    # SA.clusterSounds('tmp/', nCluster=3, descInput=[10, 5, 11])

    # SA.classifySoundkNN('tmp2/violin/56224/56224_692375-lq.json',
    #                     'tmp/', 3, descInput=[0])
    SA.classifySoundkNN('tmp2/oboe/355135/355135_6552981-lq.json',
                        'tmp/', 3, descInput=[4])

import soundDownload as SD
import soundAnalysis as SA


if __name__ == '__main__':
    # sounds = [
    #     # ('violin', 'multisample', (0, 5)),
    #     # ('guitar', 'multisample', (0, 5)),
    #     # ('bassoon', 'multisample', (0, 5)),
    #     # ('trumpet', 'multisample', (0, 5)),
    #     # ('clarinet', 'multisample', (0, 5)),
    #     # ('cello', 'multisample', (0, 5)),
    #     # ('xiaoluo', None, (0, 5)),
    #     # ('daluo', None, (0, 5)),
    #     # ('flute', 'multisample', (0, 5)),
    #     # ('mridangam', None, (0, 5))
    #     ]
    # for (text, tag, dur) in sounds:
    #     SD.downloadSoundsFreesound(queryText=text,
    #                                API_Key='kVuI0F9Qf9UbqXGYVx7TlAnF57nujpT3K9UtbCKb',
    #                                outputDir='tmp/', topNResults=20,
    #                                duration=dur, tag=tag)

    # SA.showDescriptorMapping()
    # SA.descriptorPairScatterPlot('tmp/', descInput=(0, 3))
    # SA.descriptorPairScatterPlot('tmp/', descInput=(4, 8))
    # SA.descriptorPairScatterPlot('tmp/', descInput=(10, 5))

    SA.clusterSounds('tmp/', nCluster=10, descInput=[0, 3])

    # SA.classifySoundkNN('tmp2/violin/56224/56224_692375-lq.json',
    #                     'tmp/', 3, descInput=[0])
    # SA.classifySoundkNN('tmp2/oboe/355135/355135_6552981-lq.json',
    #                     'tmp/', 3, descInput=[4])

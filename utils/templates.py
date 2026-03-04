cars_template = [
    lambda c: f'a photo of a {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a photo of my {c}.',
    lambda c: f'i love my {c}!',
    lambda c: f'a photo of my dirty {c}.',
    lambda c: f'a photo of my clean {c}.',
    lambda c: f'a photo of my new {c}.',
    lambda c: f'a photo of my old {c}.',
]

cifar10_template = [
    lambda c: f'a photo of a {c}.',
    lambda c: f'a blurry photo of a {c}.',
    lambda c: f'a black and white photo of a {c}.',
    lambda c: f'a low contrast photo of a {c}.',
    lambda c: f'a high contrast photo of a {c}.',
    lambda c: f'a bad photo of a {c}.',
    lambda c: f'a good photo of a {c}.',
    lambda c: f'a photo of a small {c}.',
    lambda c: f'a photo of a big {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a blurry photo of the {c}.',
    lambda c: f'a black and white photo of the {c}.',
    lambda c: f'a low contrast photo of the {c}.',
    lambda c: f'a high contrast photo of the {c}.',
    lambda c: f'a bad photo of the {c}.',
    lambda c: f'a good photo of the {c}.',
    lambda c: f'a photo of the small {c}.',
    lambda c: f'a photo of the big {c}.',
]

cifar100_template = [
    lambda c: f'a photo of a {c}.',
    lambda c: f'a blurry photo of a {c}.',
    lambda c: f'a black and white photo of a {c}.',
    lambda c: f'a low contrast photo of a {c}.',
    lambda c: f'a high contrast photo of a {c}.',
    lambda c: f'a bad photo of a {c}.',
    lambda c: f'a good photo of a {c}.',
    lambda c: f'a photo of a small {c}.',
    lambda c: f'a photo of a big {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a blurry photo of the {c}.',
    lambda c: f'a black and white photo of the {c}.',
    lambda c: f'a low contrast photo of the {c}.',
    lambda c: f'a high contrast photo of the {c}.',
    lambda c: f'a bad photo of the {c}.',
    lambda c: f'a good photo of the {c}.',
    lambda c: f'a photo of the small {c}.',
    lambda c: f'a photo of the big {c}.',
]

dtd_template = [
    lambda c: f'a photo of a {c} texture.',
    lambda c: f'a photo of a {c} pattern.',
    lambda c: f'a photo of a {c} thing.',
    lambda c: f'a photo of a {c} object.',
    lambda c: f'a photo of the {c} texture.',
    lambda c: f'a photo of the {c} pattern.',
    lambda c: f'a photo of the {c} thing.',
    lambda c: f'a photo of the {c} object.',
]

eurosat_template = [
    lambda c: f'a centered satellite photo of {c}.',
    lambda c: f'a centered satellite photo of a {c}.',
    lambda c: f'a centered satellite photo of the {c}.',
]

food101_template = [
    lambda c: f'a photo of {c}, a type of food.',
]

gtsrb_template = [
    lambda c: f'a zoomed in photo of a "{c}" traffic sign.',
    lambda c: f'a centered photo of a "{c}" traffic sign.',
    lambda c: f'a close up photo of a "{c}" traffic sign.',
]

mnist_template = [
    lambda c: f'a photo of the number: "{c}".',
]

imagenet_template = [
    lambda c: f'a photo of a {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a picture of a {c}.',
    lambda c: f'an image of a {c}.',
    lambda c: f'a good photo of a {c}.',
    lambda c: f'a clear photo of a {c}.',
    lambda c: f'a close-up photo of a {c}.',
    lambda c: f'a typical photo of a {c}.',
    lambda c: f'this is a {c}.',
    lambda c: f'there is a {c}.',
]

resisc45_template = [
    lambda c: f'satellite imagery of {c}.',
    lambda c: f'aerial imagery of {c}.',
    lambda c: f'satellite photo of {c}.',
    lambda c: f'aerial photo of {c}.',
    lambda c: f'satellite view of {c}.',
    lambda c: f'aerial view of {c}.',
    lambda c: f'satellite imagery of a {c}.',
    lambda c: f'aerial imagery of a {c}.',
    lambda c: f'satellite photo of a {c}.',
    lambda c: f'aerial photo of a {c}.',
    lambda c: f'satellite view of a {c}.',
    lambda c: f'aerial view of a {c}.',
    lambda c: f'satellite imagery of the {c}.',
    lambda c: f'aerial imagery of the {c}.',
    lambda c: f'satellite photo of the {c}.',
    lambda c: f'aerial photo of the {c}.',
    lambda c: f'satellite view of the {c}.',
    lambda c: f'aerial view of the {c}.',
]

stl10_template = [
    lambda c: f'a photo of a {c}.',
    lambda c: f'a photo of the {c}.',
]

sun397_template = [
    lambda c: f'a photo of a {c}.',
    lambda c: f'a photo of the {c}.',
]

svhn_template = [
    lambda c: f'a photo of the number: "{c}".',
]

kitti_template = [
    lambda c: f'a photo of a {c} from a driving perspective.',
    lambda c: f'a road scene containing a {c}.',
    lambda c: f'a street view image of a {c}.',
    lambda c: f'a driving camera view of a {c}.',
    lambda c: f'a photo of a {c} on the road.',
    lambda c: f'a photo of a {c} in traffic.',
    lambda c: f'a clear view of a {c} from a vehicle camera.',
    lambda c: f'a {c} captured by an autonomous vehicle camera.',
    lambda c: f'a photo of the {c} from a car camera.',
    lambda c: f'a dashcam photo of a {c}.',
]

vggsound_template = [
    lambda c: f'a sound of {c}.',
    lambda c: f'audio of {c}.',
    lambda c: f'the sound of {c}.',
    lambda c: f'audio recording of {c}.',
]

speech_commands_template = [
    lambda c: f'a person saying {c}.',
    lambda c: f'someone saying the word {c}.',
    lambda c: f'audio of someone saying {c}.',
    lambda c: f'a voice saying {c}.',
    lambda c: f'speech audio of {c}.',
    lambda c: f'a recording of someone saying {c}.',
    lambda c: f'spoken word {c}.',
    lambda c: f'a clear voice saying {c}.',
    lambda c: f'audio recording of {c}.',
]

epic_sounds_template = [
    lambda c: f'a sound of {c}.',
    lambda c: f'audio of {c}.',
    lambda c: f'the sound of {c}.',
    lambda c: f'kitchen sound of {c}.',
    lambda c: f'a sound of {c} in the kitchen.',
    lambda c: f'audio recording of {c}.',
    lambda c: f'household sound of {c}.',
    lambda c: f'domestic sound of {c}.',
    lambda c: f'kitchen activity sound of {c}.',
    lambda c: f'a sound of {c} during cooking.',
    lambda c: f'daily kitchen sound of {c}.',
    lambda c: f'home cooking sound of {c}.',
]

nsynth_template = [
    lambda c: f'a sound of a {c}.',
    lambda c: f'audio of a {c}.',
    lambda c: f'the sound of a {c}.',
    lambda c: f'an instrumental sound of a {c}.',
    lambda c: f'a musical sound of a {c}.',
    lambda c: f'a sound produced by a {c}.',
    lambda c: f'a recording of a {c}.',
    lambda c: f'a synthesized sound of a {c}.',
    lambda c: f'a digital sound of a {c}.',
]

audioset_template = [
    lambda c: f'a sound of {c}.',
    lambda c: f'audio of {c}.',
    lambda c: f'the sound of {c}.',
    lambda c: f'an audio clip of {c}.',
    lambda c: f'a recording of {c}.',
    lambda c: f'a sound recording of {c}.',
    lambda c: f'an audio recording of {c}.',
    lambda c: f'a sound sample of {c}.',
    lambda c: f'an audio sample of {c}.',
]

dataset_to_template = {
    'Cars': cars_template,
    'CIFAR10': cifar10_template,
    'CIFAR100': cifar100_template,
    'DTD': dtd_template,
    'EuroSAT': eurosat_template,
    'Food101': food101_template,
    'GTSRB': gtsrb_template,
    'MNIST': mnist_template,
    'imagenet': imagenet_template,
    'RESISC45': resisc45_template,
    'STL10': stl10_template,
    'SUN397': sun397_template,
    'SVHN': svhn_template,
    'KITTI': kitti_template,
    'audioset': audioset_template,
    'vggsound': vggsound_template,
    'speechcommand': speech_commands_template,
    'epicsound': epic_sounds_template,
    'nsynth': nsynth_template,
}


def get_templates(dataset_name):
    if dataset_name.endswith('Val'):
        return get_templates(dataset_name.replace('Val', ''))
    assert dataset_name in dataset_to_template, f'Unsupported dataset: {dataset_name}'
    return dataset_to_template[dataset_name]
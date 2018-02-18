import os
import random

AWS_URL = 'https://s3-us-west-1.amazonaws.com/plankton-phase2'
DB_DIR = '/data5/Plankton_wi18/rawcolor_db/'

if __name__ == '__main__':
    # Load specimen labels
    specimen_labels = [l.split('\t') for l in open(DB_DIR+'/classes/specimen_taxonomy.txt').read().splitlines()[1:]]
    specimen_labels = {lbl[0]: lbl[1:] for lbl in specimen_labels if not lbl[0].startswith('google')}
    specimen_list = os.listdir(DB_DIR+'/images')
    spc_images, spc_example = {}, {}
    corresp = {l.split()[1].replace('/', '_'): l.split()[0] for l in open('turk_db_correspondance.lst').read().splitlines()}

    # Assign example images to specimens
    examples = {}
    for spc in specimen_list:
        cls = specimen_labels[spc][2].split()[0]
        if cls == 'Unknown':
            continue
        if cls in ['Acartiidae', 'Appendicularia', 'Copepoda', 'Cydippida', 'Diphyidae', 'Euphausiidae', 'Gammaridae',
                   'Mysida', 'Oithonidae', 'Ostracoda', 'Poecilostomatoida', 'Pontellidae', 'Sergestiidae', 'Saggittoidea',
                   'Tortanus']:
            if cls == 'Diphyidae':
                cls = 'Diphyinae'
            if cls == 'Tortanus':
                cls = 'Toranidae'
            examples[spc] = cls + '.png'
        elif cls == 'Brachyura':
            if corresp[spc] == 'Brachyura_specimen05':
                examples[spc] = 'Brachyura_specimen05.png'
            elif corresp[spc] == 'Brachyura_specimen06':
                examples[spc] = 'Brachyura_specimen06.png'
            else:
                examples[spc] = 'Brachyura.png'
        elif cls == 'Calanus':
            if corresp[spc] == 'Calanidae_specimen00':
                examples[spc] = 'Calanidae_specimen00.png'
            elif corresp[spc] == 'Calanidae_specimen01':
                examples[spc] = 'Calanidae_specimen01.png'
        elif cls == 'Calanoida':
            if corresp[spc] == 'Calanoida_specimen11':
                examples[spc] = 'Calanoida_specimen11.png'
            else:
                examples[spc] = 'Calanoida.png'
        elif corresp[spc] == 'Chordata_specimen00':
            examples[spc] = 'Chordata_specimen00.png'
        elif cls == 'Hydromedusae':
            if corresp[spc] in ['Hydromedusae_specimen{:02d}'.format(i) for i in [0, 1, 2, 3, 8, 9, 10, 11, 12]]:
                examples[spc] = 'Hydromedusae_specimen00-03_08-12.png'
            elif corresp[spc] in ['Hydromedusae_specimen{:02d}'.format(i) for i in [4, 5, 6, 7]]:
                examples[spc] = 'Hydromedusae_specimen4-7.png'
        elif cls == 'Polychaeta':
            if corresp[spc] in ['Polychaeta_specimen{:02d}'.format(i) for i in [0, 1, 11]]:
                examples[spc] = 'Polychaeta_specimen00-01_11.png'
            elif corresp[spc] == 'Polychaeta_specimen02':
                examples[spc] = 'Polychaeta_specimen02.png'
            elif corresp[spc] in ['Polychaeta_specimen{:02d}'.format(i) for i in [3, 4, 5, 6, 8]]:
                examples[spc] = 'Polychaeta_specimen03-06_08.png'
            elif corresp[spc] in ['Polychaeta_specimen{:02d}'.format(i) for i in [7, 10]]:
                examples[spc] = 'Polychaeta_specimen07_10.png'
            elif corresp[spc] == 'Polychaeta_specimen09':
                examples[spc] = 'Polychaeta_specimen09.png'
        assert os.path.exists('examples/' + examples[spc])

    # Find images for each specimen and prepare links for CSV file
    all_rows = []
    for spc in specimen_list:
        cls = specimen_labels[spc][2].split()[0]
        if cls == 'Unknown':    # Ignore unlabeled data for now
            continue

        # Load list of image filenames
        image_set_fn = DB_DIR+'/subsets/{}-timeseries.lst'.format(spc)
        spc_images[spc] = open(image_set_fn).read().splitlines()
        random.shuffle(spc_images[spc])     # Randomize image order

        # Split into batches of 10 images (ignore a few images for simplicity)
        for i in range(len(spc_images[spc]) / 10):
            # Each row of CSV file should have the following format:
            #   EXAMPLE_URL, HEAD-1-URL, TAIL-1-URL, HEAD-2-URL, TAIL-2-URL, ..., HEAD-10-URL, TAIL-10-URL
            row = ['"{}/examples/{}"'.format(AWS_URL, examples[spc])]
            row += ['"{}/{}"'.format(AWS_URL, spc_images[spc][j]) for j in range(i * 10, (i + 1) * 10) for part in ('head', 'tail')]
            all_rows.append(row)
        print spc, cls, len(spc_images[spc]), examples[spc]

    # Write CSV files (split into two batch files with less then 5000 rows per batch)
    header = ['"example_img"', ] + ['"img_url{}-{}"'.format(i + 1, part) for i in range(10) for part in ('head', 'tail')]
    with open('batches/batch1.csv', 'w') as f:
        f.write(','.join(header) + '\n')
        for row in all_rows[:4800]:
            f.write(','.join(row) + '\n')

    with open('batches/batch2.csv', 'w') as f:
        f.write(','.join(header) + '\n')
        for row in all_rows[4800:]:
            f.write(','.join(row) + '\n')
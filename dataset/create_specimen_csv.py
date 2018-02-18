import utils
import glob

if __name__ == '__main__':
    root = utils.plankton_taxonomy()

    # Print specimen list
    print '\n', '='*20, 'Specimen List', '='*20
    f = open('specimen_db.csv', 'w')
    f.write('SpecimenID,Image Count,Order,Family,Genus,URL,Comments\n')
    nodes = [root]
    while nodes:
        n = nodes.pop(0)
        if len(n['children']) == 0:
            for spc in n['specimen_list']:
                if spc.endswith('Bad'):
                    continue
                image_count = len(glob.glob('/data4/plankton_wi17/plankton/images_orig/{}/0000000_static_html/images/*/*rawcolor.png'.format(spc.replace('_', '/'))))
                genus = n['name']
                family = n['parent']['name']
                order = n['parent']['parent']['name']
                url = 'http://www.svcl.ucsd.edu/~morgado/plankton/{}/0000000_static_html/spcdata.html'.format(spc.replace('_', '/'))
                f.write('"{}","{}","{}","{}","{}","{}",\n'.format(spc, image_count, order, family, genus, url))
                print '"{}","{}","{}","{}","{}","{}"'.format(spc, image_count, order, family, genus, url)
        else:
            nodes.extend(n['children'])
    f.close()
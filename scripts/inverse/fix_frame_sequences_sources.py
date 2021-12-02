from wormlab3d import logger
from wormlab3d.data.model import FrameSequence


def fix_FSs():
    FSs = FrameSequence.objects
    N = FSs.count()
    logger.info(f'Found {N} frame sequences.')
    i = 1
    for FS in FSs:
        logger.info(f'Checking FS {i}/{N} id={FS.id}.')
        if FS.source is None:
            logger.info('Missing source! Checking midlines..')
            sources = []
            source_files = []
            for midline in FS.midlines:
                sources.append(midline.source)
                source_files.append(midline.source_file)
            sources_unique = list(set(sources))
            source_files_unique = list(set(source_files))

            if len(sources_unique) == 1 and len(set(source_files_unique)) == 1:
                source = sources_unique[0]
                source_file = source_files_unique[0]
                logger.info(f'Setting source={source} and source_file={source_file}')
                FS.source = source
                FS.source_file = source_file
                FS.save()
            else:
                if len(sources_unique) != 1:
                    logger.warning(f'Found multiple sources: {sources_unique}!')
                if len(source_files_unique) != 1:
                    logger.warning(f'Found multiple source files: {source_files_unique}!')
        i += 1


if __name__=='__main__':
    fix_FSs()

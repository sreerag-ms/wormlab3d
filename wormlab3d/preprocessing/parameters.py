from collections import namedtuple

Parameters = namedtuple( 'Parameters', [ 'pvd_dir', 'frame_dir', 'output_dir'] )

def parser( arguments ):
    pvd_dir = './'
    frame_dir = './'
    output_dir = './'

    for arg in arguments:
        d = arg.split(':')
        if d[0].strip() == 'pvd_dir':
            pvd_dir = d[1].strip()
            print('pvd_dir',':',pvd_dir)
        if d[0].strip() == 'frame_dir':
            frame_dir = d[1].strip()
            print('frame_dir',':',frame_dir)
        if d[0].strip() == 'output_dir':
            output_dir = d[1].strip()
            print('output_dir',':',output_dir)

    if pvd_dir == './':
        print('pvd_dir',':',pvd_dir,'(default)')
    if frame_dir == './':
        print('frame_dir',':',frame_dir,'(default)')
    if output_dir == './':
        print('output_dir',':',output_dir,'(default)')

    return Parameters( pvd_dir=pvd_dir,
                       frame_dir=frame_dir,
                       output_dir=output_dir )

def parserDict( arguments ):
    ret = {}
    for arg in arguments:
        if ':' in arg:
            try:
                d = arg.split(':')
                key = d[0].strip()
                value = d[1].strip()
                ret[key] = value
                print(key,':',value)
            except:
                print('unable to read', arg)

    return ret


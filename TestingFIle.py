import sys
import argparse
import mpyq
from s2protocol import versions


def read_contents(archive, content):
    contents = archive.read_file(content)
    if not contents:
        print('Error: Archive missing {}'.format(content))
        sys.exit(1)
    return contents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('replay_file', help='.SC2Replay file to load', nargs='?')
    args = parser.parse_args()

    # Check/test the replay file
    if args.replay_file is None:
        print(sys.stderr, ".S2Replay file not specified")
        sys.exit(1)

    archive = mpyq.MPQArchive(args.replay_file)

    # HEADER
    # contents = archive.header['user_data_header']['content']
    # header = versions.latest().decode_replay_header(contents)

    contents = read_contents(archive, 'replay.game.events')
    details = versions.latest().decode_replay_game_events(contents)

    for x in details['m_playerList']:
        print('hello')
        #print({
        #    'name': x['m_name'],
        #    'handle': '%d-S%d-%d-%d' % (x['m_toon']['m_region'], x['m_toon']['m_region'], x['m_toon']['m_realm'], x['m_toon']['m_id']),
        #})


    #contents = read_contents(archive, 'replay.details')
    #details = versions.latest().decode_replay_details(contents)
    #for x in details['m_playerList']:
    #    print({
    #        'name': x['m_name'],
    #        'handle': '%d-S%d-%d-%d' % (x['m_toon']['m_region'], x['m_toon']['m_region'], x['m_toon']['m_realm'], x['m_toon']['m_id']),
    #    })

if __name__ == '__main__':
    main()

#s2_cli.main("D:\SC2_ReplayData\Replays\b6e865eb7c096cd682eb036be69753c47d7c70a6aef2271be45c940143e9b0e5.SC2REPLAY")
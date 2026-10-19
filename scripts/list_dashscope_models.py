"""List models advertised by the configured endpoint without generating text."""
import argparse
import os
from urllib.parse import urlparse

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='https://dashscope.aliyuncs.com/compatible-mode/v1')
    args = parser.parse_args()
    key = os.environ.get('DASHSCOPE_API_KEY', '').strip()
    if not key:
        parser.error('Set DASHSCOPE_API_KEY in this terminal first.')
    if urlparse(args.base_url).scheme != 'https':
        parser.error('Use an HTTPS endpoint.')
    url = args.base_url.rstrip('/') + '/models'
    print('Endpoint:', url)
    try:
        response = requests.get(
            url, headers={'Authorization': 'Bearer ' + key}, timeout=30,
            allow_redirects=False,
        )
    except requests.RequestException as exc:
        raise SystemExit('Request failed: ' + str(exc).replace(key, '<redacted>'))
    if response.status_code != 200:
        print('HTTP', response.status_code)
        print(response.text.replace(key, '<redacted>')[:3000])
        raise SystemExit('Listing failed. This endpoint may not support /models; check the console API example.')
    try:
        data = response.json()['data']
        if not isinstance(data, list):
            raise ValueError('data is not a list')
        models = sorted({row['id'] for row in data if isinstance(row, dict) and isinstance(row.get('id'), str)})
    except (ValueError, KeyError, TypeError):
        raise SystemExit('Unexpected model-list response; check the console API example.')
    print('\nAll advertised models ({}):'.format(len(models)))
    for model in models:
        print('  ' + model.replace(key, '<redacted>'))
    candidates = [m for m in models if any(tag in m.lower() for tag in ('vl', 'vision', 'omni'))]
    print('\nPossible vision models (name-based filter only):')
    for model in candidates:
        print('  ' + model.replace(key, '<redacted>'))
    if not candidates:
        print('  None identified by name.')
    print('\nA listed model is not proof of image support or account access.')
    print('Confirm with one image request before starting the experiment.')
    print('No generation requests were made.')


if __name__ == '__main__':
    main()

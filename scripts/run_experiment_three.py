"""Run only the revision auxiliary-loss ablation on both sites."""
import sys
from run_revision_followups import main

if __name__ == '__main__':
    sys.argv.insert(1, '3')
    main()

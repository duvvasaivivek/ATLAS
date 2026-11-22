import unittest
from greengpu.timing import breakdown


class TimingTests(unittest.TestCase):
    def test_shared_prefix_is_not_counted_twice_or_used_as_compute(self):
        r=breakdown(dict(release_ns=0,communication_end_ns=100,cpu_end_ns=110,
                         gpu_end_ns=120,end_ns=123),'hotspot')
        self.assertEqual(r['shared_communication_ns'],100)
        self.assertEqual(r['cpu_after_shared_ns'],10)
        self.assertEqual(sum(r[k] for k in ('shared_communication_ns','parallel_envelope_ns','join_tail_ns','merge_ns')),123)
        self.assertEqual(r['branch_wait_opportunity_ns'],10)

    def test_empty_branch_and_merge(self):
        r=breakdown(dict(release_ns=0,cpu_end_ns=None,gpu_end_ns=20,barrier_end_ns=22,end_ns=30),'kmeans')
        self.assertIsNone(r['cpu_completion_ns'])
        self.assertEqual(r['merge_ns'],8)
        self.assertEqual(r['join_tail_ns'],2)

"""Stage implementations for the M1–M5 diagnostic pipeline.

M1  probe.py / probe_agent.py / protocol.py   — analyzer selection + execution
M2  analysis.py / stats_agent.py              — statistical interpretation
M3  diagnosis.py                              — hypothesis generation (AI scientist)
M4  surgery.py / experiment_writer.py         — fix proposal + execution
M5  hypothesis_tester.py                      — statistical + protocol consistency test
"""

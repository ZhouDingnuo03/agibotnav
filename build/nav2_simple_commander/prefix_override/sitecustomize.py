import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/orin-001/sda/agibotnav/install/nav2_simple_commander'

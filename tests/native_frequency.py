"""Independent Python oracle including small and partial CUDA blocks."""
import json
import subprocess
import sys

binary=sys.argv[1]
for mode in ('compute','stream'):
    for n in (1,257,4099):
        rounds=17 if mode=='compute' else 1
        result=subprocess.run([binary,mode,str(n),str(rounds),'1729'],input='run 3\nrun 2\nquit\n',text=True,capture_output=True,timeout=30)
        assert result.returncode==0,result.stderr
        records=[json.loads(s) for s in result.stdout.splitlines()]
        checksum=0
        for i in range(n):
            x=i^1729
            if mode=='compute':
                for _ in range(rounds):
                    x=(x^(x<<13))&0xffffffff;x=(x^(x>>17))&0xffffffff;x=(x^(x<<5))&0xffffffff
            else:x=(x*1664525+1013904223)&0xffffffff
            checksum+=x
        assert len(records)==3 and records[0]['kind']=='ready'
        for i,r in enumerate(records[1:]):
            assert r['batch']==i and r['checksum']==checksum and r['mismatches']==0
            assert r['elements_verified']==n and r['repetitions']==(3 if i==0 else 2)
            assert r['start_ns']<=r['compute_end_ns']<=r['download_end_ns']<=r['validation_end_ns']
for args,stdin in [(['bad','1','1','1'],'quit\n'),(['stream','1','2','1'],'quit\n'),(['compute','257','1','1'],'run 0\n')]:
    result=subprocess.run([binary,*args],input=stdin,text=True,capture_output=True,timeout=30)
    assert result.returncode!=0
print('Two diagnostics, six sizes/modes, twelve batches match independent Python checksums; invalid inputs rejected.')

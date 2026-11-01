from optimization_problem import CureProblemDefinition,pareto_front,hypervolume_2d,representative_solutions

def fake(t,a):
 return CureProblemDefinition().analyze(dict(hold_end_min=t,alpha_min=a,peak_T_C=191.,alpha_spread=.02))
r1=fake(250,.81);r2=fake(240,.82);r3=fake(250,.83);r4=fake(260,.825)
assert not r1['feasible'] and r2['feasible']
assert r2['F']==[240,-.82]
f=pareto_front([r1,r2,r3,r4]);assert len(f)==2
assert f==[r2,r3]
assert abs(hypervolume_2d([r2])-65.6)<1e-9
assert hypervolume_2d([r1])==0
assert representative_solutions([r1,r2,r3])['shortest_time']==r2
print('PASS: objectives, constraint, Pareto, hypervolume, representatives')

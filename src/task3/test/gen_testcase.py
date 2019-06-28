import sys
import numpy as np

testtypes = ['aggregate1', 'aggregate2', 'readout', 'gnn_call', 'linear']

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('python gen_testcase.py [testtype] [filename]')
        exit(1)
    
    testtype = sys.argv[1]
    filename = sys.argv[2]
    if testtype not in testtypes:
        print('Invalid testtype', testtype)
        exit(1)

    if testtype == 'aggregate1':
        n = np.random.randint(2, 21)
        d = np.random.randint(1, 101)

        graph = np.random.randint(2, size=[n, n])
        graph = np.tril(graph, -1) + np.tril(graph, -1).T
        x = 2*np.random.rand(n, d) - 1
        ans = np.zeros([n, d])
        
        for v in range(n):
            for w in range(n):
                if graph[w, v] != 1:
                    continue
                ans[v] += x[w]
        
        with open(filename, 'w') as o:
            o.write('%d %d\n' % (n, d))
            for i in range(n):
                o.write(' '.join(map(str, graph[i])) + '\n')
            for i in range(n):
                o.write(' '.join(map(str, x[i])) + '\n')
            for i in range(n):
                o.write(' '.join(map(str, ans[i])) + '\n')

    elif testtype == 'aggregate2':
        n = np.random.randint(2, 21)
        d = np.random.randint(1, 101)

        W = 2*np.random.rand(d, d) - 1
        x = 2*np.random.rand(n, d) - 1
        ans = np.zeros([n, d])
        
        for v in range(n):
            for i in range(d):
                ans[v, i] = max(0, np.sum(W[i]*x[v]))

        with open(filename, 'w') as o:
            o.write('%d %d\n' % (n, d))
            for i in range(d):
                o.write(' '.join(map(str, W[i])) + '\n')
            for i in range(n):
                o.write(' '.join(map(str, x[i])) + '\n')
            for i in range(n):
                o.write(' '.join(map(str, ans[i])) + '\n')

    elif testtype == 'readout':
        n = np.random.randint(2, 21)
        d = np.random.randint(1, 101)
        x = 2*np.random.rand(n, d) - 1
        ans = np.zeros([d])
        
        for i in range(d):
            ans[i] = sum(x[:, i])

        with open(filename, 'w') as o:
            o.write('%d %d\n' % (n, d))
            for i in range(n):
                o.write(' '.join(map(str, x[i])) + '\n')
            o.write(' '.join(map(str, ans)) + '\n')

    elif testtype == 'linear':
        p, q, r = np.random.randint(1, 21, size=3)
        W = 2*np.random.rand(p, q) - 1
        b = 2*np.random.rand(p) - 1
        x = 2*np.random.rand(r, q) - 1
        ans = np.zeros([r, p])

        for i in range(r):
            for j in range(p):
                for k in range(q):
                    ans[i, j] += W[j, k]*x[i, k]
                ans[i, j] += b[j]
        
        with open(filename, 'w') as o:
            o.write('%d %d %d\n' % (p, q, r))
            for i in range(p):
                o.write(' '.join(map(str, W[i])) + '\n')
            o.write(' '.join(map(str, b)) + '\n')
            for i in range(r):
                o.write(' '.join(map(str, x[i])) + '\n')
            for i in range(r):
                o.write(' '.join(map(str, ans[i])) + '\n')

    elif testtype == 'gnn_call':
        t = np.random.randint(1, 11)
        n = np.random.randint(2, 21)
        d = np.random.randint(1, 101)
        W = 2*np.random.rand(d, d) - 1
        graph = np.random.randint(2, size=[n, n])
        graph = np.tril(graph, -1) + np.tril(graph, -1).T
        x = 2*np.random.rand(n, d) - 1
        A = 2*np.random.rand(d) - 1
        b = 2*np.random.rand() - 1

        h = x
        for _ in range(t):
            # aggregate1
            anxt = np.zeros_like(h)
            for v in range(n):
                for w in range(n):
                    if graph[w, v] != 1:
                        continue
                    anxt[v] += h[w]
            # aggregate2
            hnxt = np.zeros_like(h) 
            for v in range(n):
                for i in range(d):
                    hnxt[v, i] = max(0, np.sum(W[i]*anxt[v]))
    
            h = hnxt

        # readout
        s = np.zeros([d])
        for i in range(d):
            s[i] = sum(h[:, i])

        # linear
        ans = 0
        for i in range(d):
            ans += A[i]*s[i]
        ans += b

        with open(filename, 'w') as o:
            o.write('%d %d %d\n' % (t, n, d))
            for i in range(d):
                o.write(' '.join(map(str, W[i])) + '\n')
            for i in range(n):
                o.write(' '.join(map(str, graph[i])) + '\n')
            for i in range(n):
                o.write(' '.join(map(str, x[i])) + '\n')
            o.write(' '.join(map(str, A)) + '\n')
            o.write(str(b) + '\n')
            o.write(str(ans) + '\n')

    else:
        print('Internal bug')

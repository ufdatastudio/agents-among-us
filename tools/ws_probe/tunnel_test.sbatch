#!/bin/bash
#SBATCH --job-name=aau-tunnel
#SBATCH --account=ufdatastudios
#SBATCH --qos=ufdatastudios
#SBATCH --time=00:14:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1gb
#SBATCH --output=%x_%j.out
# Submit from a directory that holds ws_probe_server.py and a known_hosts file with the host key of the VM.
# The key ~/.ssh/aau_tunnel_test is created for the test and authorized on the VM with
#   restrict,port-forwarding,permitlisten="127.0.0.1:19080",command="/bin/false"
# and both copies are removed afterwards.
# Tests whether a HiPerGator compute node can open a reverse SSH tunnel to the PubApps VM and serve WebSockets through it.
DIR=$SLURM_SUBMIT_DIR
PORT=19080
echo "node $(hostname), started $(date +%T), python $(python3 --version 2>&1)"
python3 $DIR/ws_probe_server.py --port $PORT --ttl 780 &
SERVER=$!
sleep 2
ssh -N -v -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -o IdentitiesOnly=yes -i $HOME/.ssh/aau_tunnel_test \
  -o UserKnownHostsFile=$DIR/known_hosts -o StrictHostKeyChecking=yes \
  -R 127.0.0.1:$PORT:127.0.0.1:$PORT ufdatastudios@pubufdatastudios1 2> $DIR/ssh_tunnel.log &
TUNNEL=$!
echo "server pid $SERVER, tunnel pid $TUNNEL"
sleep 8
if kill -0 $TUNNEL 2>/dev/null; then echo "tunnel is up at $(date +%T)"; else echo "tunnel failed at $(date +%T)"; tail -20 $DIR/ssh_tunnel.log; kill $SERVER; exit 1; fi
wait $SERVER
kill $TUNNEL 2>/dev/null
echo "finished $(date +%T)"

#!/bin/bash
# Vigilante de memoria para pruebas de diagnóstico.
#
#   tools/vigilar_memoria.sh <limite_GB> <limite_seg> <log> -- <comando...>
#
# Lanza el comando, muestrea cada segundo la memoria residente de TODO su árbol de
# procesos y lo mata entero si supera el límite de memoria o de tiempo. Al terminar
# imprime el pico de memoria, la duración y el motivo. Sirve para probar métodos o
# configuraciones nuevas sin riesgo de saturar la máquina.
LIM_GB=$1; LIM_S=$2; LOG=$3; shift 4
"$@" > "$LOG" 2>&1 &
PID=$!
PEAK=0; T0=$(date +%s); MOTIVO="terminó normalmente"

arbol() {  # PIDs del árbol de procesos de $1 (incluido)
  local todos=$1 nuevos=$1
  while [ -n "$nuevos" ]; do
    nuevos=$(ps -A -o pid=,ppid= | awk -v p=" $nuevos " 'index(p, " "$2" ") {print $1}' | tr '\n' ' ')
    todos="$todos $nuevos"
  done
  echo $todos
}

while kill -0 $PID 2>/dev/null; do
  PIDS=$(arbol $PID | tr ' ' ',' | sed 's/,,*/,/g; s/^,//; s/,$//')
  RSS=$(ps -o rss= -p "$PIDS" 2>/dev/null | awk '{s+=$1} END {print s+0}')
  [ "$RSS" -gt "$PEAK" ] && PEAK=$RSS
  EL=$(( $(date +%s) - T0 ))
  if [ $(( RSS / 1048576 )) -ge "$LIM_GB" ] || [ "$EL" -ge "$LIM_S" ]; then
    [ $(( RSS / 1048576 )) -ge "$LIM_GB" ] && MOTIVO="MATADO: memoria >= ${LIM_GB} GB" || MOTIVO="MATADO: tiempo >= ${LIM_S} s"
    kill -9 $(arbol $PID) 2>/dev/null
    break
  fi
  sleep 1
done
wait $PID 2>/dev/null
echo "pico_memoria_GB=$(awk -v k=$PEAK 'BEGIN{printf "%.1f", k/1048576}')  tiempo_s=$(( $(date +%s) - T0 ))  resultado: $MOTIVO"

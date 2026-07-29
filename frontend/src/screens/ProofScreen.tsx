import { useEffect, useRef, useState } from 'react'

type MicStatus = 'idle' | 'requesting' | 'active' | 'denied'

function ProofScreen() {
  const [micStatus, setMicStatus] = useState<MicStatus>('idle')
  const [micLevel, setMicLevel] = useState(0)

  const audioContextRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const rafRef = useRef<number | null>(null)

  const getAudioContext = () => {
    if (!audioContextRef.current) {
      audioContextRef.current = new AudioContext()
    }
    return audioContextRef.current
  }

  const playViaAudioElement = () => {
    const audio = new Audio('/sample.wav')
    void audio.play()
  }

  const playViaWebAudioApi = async () => {
    const context = getAudioContext()
    const response = await fetch('/sample.wav')
    const arrayBuffer = await response.arrayBuffer()
    const audioBuffer = await context.decodeAudioData(arrayBuffer)
    const source = context.createBufferSource()
    source.buffer = audioBuffer
    source.connect(context.destination)
    source.start()
  }

  const startMicTest = async () => {
    setMicStatus('requesting')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream

      const context = getAudioContext()
      const analyser = context.createAnalyser()
      analyser.fftSize = 2048
      const source = context.createMediaStreamSource(stream)
      source.connect(analyser)
      analyserRef.current = analyser

      setMicStatus('active')

      const dataArray = new Uint8Array(analyser.fftSize)
      const tick = () => {
        analyser.getByteTimeDomainData(dataArray)
        let sumSquares = 0
        for (const sample of dataArray) {
          const centered = (sample - 128) / 128
          sumSquares += centered * centered
        }
        const rms = Math.sqrt(sumSquares / dataArray.length)
        setMicLevel(rms)
        rafRef.current = requestAnimationFrame(tick)
      }
      tick()
    } catch {
      setMicStatus('denied')
    }
  }

  const stopMicTest = () => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    analyserRef.current = null
    setMicStatus('idle')
    setMicLevel(0)
  }

  useEffect(() => stopMicTest, [])

  return (
    <main className="game-screen">
      <p className="status-message">
        Proves mic access and audio playback both work in the browser, independent of any game
        logic.
      </p>

      <section className="panel">
        <h2>Playback proof</h2>
        <div className="button-row">
          <button type="button" className="btn btn-secondary" onClick={playViaAudioElement}>
            Play via &lt;audio&gt;
          </button>
          <button type="button" className="btn btn-secondary" onClick={() => void playViaWebAudioApi()}>
            Play via Web Audio API
          </button>
        </div>
      </section>

      <section className="panel">
        <h2>Mic proof</h2>
        {micStatus === 'idle' && (
          <button type="button" className="btn btn-primary" onClick={() => void startMicTest()}>
            Test Mic
          </button>
        )}
        {micStatus === 'requesting' && <p className="muted">Requesting mic permission&hellip;</p>}
        {micStatus === 'denied' && (
          <p className="form-error" role="alert">
            Mic permission denied or unavailable.
          </p>
        )}
        {micStatus === 'active' && (
          <>
            <div className="mic-meter">
              <div
                className="mic-meter-fill"
                style={{ width: `${Math.min(micLevel * 400, 100)}%` }}
              />
            </div>
            <button type="button" className="btn btn-secondary" onClick={stopMicTest} style={{ marginTop: '0.75rem' }}>
              Stop
            </button>
          </>
        )}
      </section>
    </main>
  )
}

export default ProofScreen

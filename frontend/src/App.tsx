import GameScreen from './components/GameScreen'
import ProofScreen from './screens/ProofScreen'

function App() {
  const screen = new URLSearchParams(window.location.search).get('screen')
  return screen === 'proof' ? <ProofScreen /> : <GameScreen />
}

export default App

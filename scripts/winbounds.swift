// winbounds <pid> [--front] — on-screen windows of one process, for scripts/fx_gate.py.
// Prints "<windowNumber> <x> <y> <w> <h> <layer>" per window (points, top-left origin).
// --front first tries to activate the process (cooperative activation may refuse).
import AppKit
import CoreGraphics

let pid = pid_t(CommandLine.arguments[1])!
if CommandLine.arguments.contains("--front"), let app = NSRunningApplication(processIdentifier: pid) {
    app.activate(options: [.activateAllWindows])
    usleep(300_000)
}
let list = CGWindowListCopyWindowInfo([.optionOnScreenOnly], kCGNullWindowID) as! [[String: Any]]
for w in list {
    guard (w[kCGWindowOwnerPID as String] as? Int) == Int(pid) else { continue }
    let b = w[kCGWindowBounds as String] as! [String: Any]
    print(w[kCGWindowNumber as String]!, b["X"]!, b["Y"]!, b["Width"]!, b["Height"]!, w[kCGWindowLayer as String]!)
}

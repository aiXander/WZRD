// cgmouse — synthetic mouse input for unattended UI tests (docs/TODO/afk-test-session.md).
//   cgmouse click <x> <y>
//   cgmouse drag <x0> <y0> <x1> <y1> [steps]
// Screen points, top-left origin (same space as winbounds.swift). Moves the
// real cursor: only run while nobody is at the Mac. Posting events needs the
// runner's terminal to have Accessibility permission.
import CoreGraphics
import Foundation

func post(_ type: CGEventType, _ p: CGPoint) {
    CGEvent(mouseEventSource: nil, mouseType: type, mouseCursorPosition: p, mouseButton: .left)?
        .post(tap: .cghidEventTap)
}

let a = CommandLine.arguments
let n = a.dropFirst(2).compactMap(Double.init)
switch (a.count > 1 ? a[1] : "", n.count) {
case ("click", 2):
    let p = CGPoint(x: n[0], y: n[1])
    post(.mouseMoved, p); usleep(50_000)
    post(.leftMouseDown, p); usleep(50_000)
    post(.leftMouseUp, p)
case ("drag", 4...):
    let (p0, p1) = (CGPoint(x: n[0], y: n[1]), CGPoint(x: n[2], y: n[3]))
    let steps = max(Int(n.count > 4 ? n[4] : 20), 1)
    post(.mouseMoved, p0); usleep(50_000)
    post(.leftMouseDown, p0); usleep(80_000)
    for i in 1...steps {
        let t = Double(i) / Double(steps)
        post(.leftMouseDragged, CGPoint(x: p0.x + (p1.x - p0.x) * t, y: p0.y + (p1.y - p0.y) * t))
        usleep(16_000)
    }
    post(.leftMouseUp, p1)
default:
    FileHandle.standardError.write("usage: cgmouse click x y | cgmouse drag x0 y0 x1 y1 [steps]\n".data(using: .utf8)!)
    exit(2)
}

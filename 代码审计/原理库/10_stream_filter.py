"""闭环 10：流式过滤（把 <memory> 与 <<钩子>> 从气泡里抠掉）

跑：python 10_stream_filter.py
对应源码：brain/mind.py::StreamFilter
"""
M_OPEN, M_CLOSE = "<<", ">>"
MEM_OPEN, MEM_CLOSE = "<memory>", "</memory>"


class StreamFilter:
    def __init__(self):
        self.buf = ""
        self.markers = []
        self.memory = ""
        self.in_mem = False

    def _hold(self) -> int:
        best = 0
        for mk in (M_OPEN, MEM_OPEN):
            for k in range(len(mk) - 1, 0, -1):
                if self.buf.endswith(mk[:k]):
                    best = max(best, k)
                    break
        return best

    def feed(self, chunk: str) -> str:
        self.buf += chunk
        out = []
        while True:
            if self.in_mem:
                j = self.buf.find(MEM_CLOSE)
                if j < 0:
                    hold = len(MEM_CLOSE) - 1
                    if len(self.buf) > hold:
                        self.memory += self.buf[:-hold]
                        self.buf = self.buf[-hold:]
                    break
                self.memory += self.buf[:j]
                self.buf = self.buf[j + len(MEM_CLOSE):]
                self.in_mem = False
                continue

            i, m = self.buf.find(M_OPEN), self.buf.find(MEM_OPEN)
            if m >= 0 and (i < 0 or m < i):
                out.append(self.buf[:m])
                self.buf = self.buf[m + len(MEM_OPEN):]
                self.in_mem = True
                continue
            if i < 0:
                h = self._hold()
                if h:
                    out.append(self.buf[:-h])
                    self.buf = self.buf[-h:]
                else:
                    out.append(self.buf)
                    self.buf = ""
                break
            j = self.buf.find(M_CLOSE, i + 2)
            if j < 0:
                out.append(self.buf[:i])
                self.buf = self.buf[i:]
                break
            out.append(self.buf[:i])
            self.markers.append(self.buf[i + 2:j])
            self.buf = self.buf[j + 2:]
        return "".join(out)

    def finish(self) -> str:
        rest, self.buf = self.buf, ""
        if self.in_mem:
            return ""
        if M_OPEN in rest:
            rest = rest[:rest.find(M_OPEN)]
        return rest


if __name__ == "__main__":
    f = StreamFilter()
    chunks = ["你好", "，我记一下", "<memo", 'ry>{"a":1}', "</memory>",
              " <<after=", "120 note=过两分钟>>", "好的"]
    visible = "".join(f.feed(c) for c in chunks) + f.finish()
    print("气泡内容:", repr(visible))
    print("memory  :", f.memory)
    print("markers :", f.markers)
    assert visible == "你好，我记一下 好的", visible
    assert f.memory == '{"a":1}', f.memory
    assert f.markers == ["after=120 note=过两分钟"], f.markers
    print("OK：跨 chunk 的标记、记忆块、钩子都被正确剥离")

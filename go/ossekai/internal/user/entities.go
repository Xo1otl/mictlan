package user

import "ossekaiserver/internal/auth"

type Profile struct {
	Sub         auth.Sub
	NickName    string
	Description string
}

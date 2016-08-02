require 'torch'

lua_utils = {}

function lua_utils:secondsToClock(sSeconds)
	local nSeconds = tonumber(sSeconds)
	if nSeconds == 0 then
	--return nil;
		return "00:00:00";
	else
		nHours = string.format("%02.f", math.floor(nSeconds/3600));
		nMins = string.format("%02.f", math.floor(nSeconds/60 - (nHours*60)));
		nSecs = string.format("%02.f", math.floor(nSeconds - nHours*3600 - nMins *60));
		return nHours..":"..nMins..":"..nSecs .. ' (hh:mm:ss)'
	end
end

return lua_utils


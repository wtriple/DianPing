--获取锁中的标识，判断是否与当前线程一致
if(redis.call('GET', KEYS[1])==ARGV[1]) then
    return redis.call('DEL', KEYS[1])
end
return 0